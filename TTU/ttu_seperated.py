from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import pickle
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb


# TrainingConfig
LR = 3e-5
BATCH_SIZE = 32
EPOCHS = 4
WANDB = True

# ModelConfig
EMBED_DIM = 1024  # Embedding dimension
FFN_DIM = 2048 # Default 2048
DROPOUT = 0.1 # Default 0.1
NUM_HEADS = 8 
NUM_LAYERS = 6
SRC_MASK_TYPE = "none" # none or causal
TGT_MASK_TYPE = "causal" #"independent" or causal
MEMORY_MASK_TYPE = "none" # none, independent or causal

if SRC_MASK_TYPE == "none":
    assert MEMORY_MASK_TYPE == "none"

NUM_TEXT_TOKENS = 128_256
NUM_UNIT_TOKENS = 10_000 + 4  # Example token size including <eos> and <sos> tokens
SOS_TOKEN = NUM_UNIT_TOKENS - 3
EOS_TOKEN = NUM_UNIT_TOKENS - 4
SEP_TOKEN = NUM_UNIT_TOKENS - 1
PAD_TOKEN = NUM_UNIT_TOKENS - 2

# Load data function
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(chunks):
    if type(chunks) == int:
        chunks = [chunks]

    # x: list of chunk indices
    TUPAIR_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/chunk_{}.pkl"
    ALIGNMENT_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/result/chunk_{}.pkl"

    text_tokens = []
    unit_tokens = []
    alignment = []
    for x in chunks:
        # if x in [0, 4, 15, 23]:
        #     continue # TODO: fix this
        tupair_data = load_pickle(TUPAIR_FILE_TMPL.format(x))
        chunk_text_tokens = tupair_data['text']
        chunk_unit_tokens = tupair_data['unit']

        chunk_text_tokens = [t for t, u in zip(chunk_text_tokens, chunk_unit_tokens) if len(u) <= 1024]
        chunk_unit_tokens = [u for u in chunk_unit_tokens if len(u) <= 1024]

        text_tokens.extend(chunk_text_tokens)
        unit_tokens.extend(chunk_unit_tokens)

        alignment_data = load_pickle(ALIGNMENT_FILE_TMPL.format(x))
        alignment.extend(alignment_data)

        # import ipdb; ipdb.set_trace()

        print(x, len(alignment_data), len(chunk_text_tokens))
        assert len(alignment_data) == len(chunk_text_tokens)
        assert len(alignment_data) == len(chunk_unit_tokens)
    
    for i in range(len(text_tokens)):
        sq_len = len(text_tokens[i])
        # print(sq_len, len(alignment[i]))
        alignment[i] = alignment[i][:sq_len]

    # only first token
    new_text_tokens = []
    new_unit_tokens = []
    new_alignment = []
    for i in range(len(text_tokens)):
        cur_pos = 0
        for j in range(len(text_tokens[i])):
            new_text_tokens.append(text_tokens[i][j : j+1])
            new_unit_tokens.append(unit_tokens[i][cur_pos : cur_pos+alignment[i][j]])
            new_alignment.append(alignment[i][j : j+1])

            cur_pos += alignment[i][j]

    return new_text_tokens, new_unit_tokens, new_alignment

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, text_tokens, unit_tokens, alignment):
        self.text_tokens = text_tokens
        self.unit_tokens = unit_tokens
        self.alignment = alignment

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        text_seq = self.text_tokens[idx]
        unit_seq = self.unit_tokens[idx]
        align_len = self.alignment[idx]

        aligned_unit_seq_input = [SOS_TOKEN]
        aligned_unit_seq_y = []
        cur_pos = 0
        align_len[0] += 1
        for i, length in enumerate(align_len):
            # aligned_unit_seq_input.append(SEP_TOKEN)  # Note that we use SOS here
            aligned_unit_seq_input.extend(unit_seq[cur_pos: cur_pos + length])

            aligned_unit_seq_y.extend(unit_seq[cur_pos: cur_pos + length])
            # aligned_unit_seq_y.append(SEP_TOKEN)  # and use EOS here

            cur_pos += length
            # align_len[i] += 1
        aligned_unit_seq_y.append(EOS_TOKEN)
        align_len[-1] += 1

        return text_seq, aligned_unit_seq_input, aligned_unit_seq_y, align_len

def create_chunked_tgt_mask(tgt_seq_len, alignment_len):
    tgt_mask = torch.full((tgt_seq_len, tgt_seq_len), float('-inf'))
    current_idx = 0

    for chunk_size in alignment_len:
        for i in range(current_idx, current_idx + chunk_size):
            tgt_mask[i, current_idx: i + 1] = 0.0  # intra-chunk causal attention
            # tgt_mask[i, 0: i + 1] = 0.0
        current_idx += chunk_size

    return tgt_mask

def pad_sequences(sequences, pad_token):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_seqs)

# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_text_tokens, num_unit_tokens, emb_size, nhead, num_encoder_layers, num_decoder_layers):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, 
                                          dim_feedforward=FFN_DIM, dropout=DROPOUT,
                                          batch_first=True,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers)
        self.text_embedding = nn.Embedding(num_text_tokens, emb_size)
        self.unit_embedding = nn.Embedding(num_unit_tokens, emb_size)
        self.fc_out = nn.Linear(emb_size, num_unit_tokens)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        src_emb = self.text_embedding(src)
        tgt_emb = self.unit_embedding(tgt)
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.fc_out(outs)

def create_mask(src, tgt, alignment_lens):
    batch_size = src.size(0)
    src_seq_len = src.size(-1)
    tgt_seq_len = tgt.size(-1)

    # First generate padding masks
    src_key_padding_mask = torch.zeros((batch_size, src_seq_len))
    tgt_key_padding_mask = torch.zeros((batch_size, tgt_seq_len))
    for b in range(batch_size):
        src_len = len(alignment_lens[b])
        tgt_len = np.array(alignment_lens[b]).sum()
        src_key_padding_mask[b][src_len:] = float("-inf")
        tgt_key_padding_mask[b][tgt_len:] = float("-inf")
    memory_key_padding_mask = src_key_padding_mask.clone()
        

    if SRC_MASK_TYPE == "none":
        # still need padding mask
        src_mask = torch.zeros((src_seq_len, src_seq_len))
        
        # # debug
        # src_mask = torch.zeros((batch_size, src_seq_len, src_seq_len))
        # for b in range(batch_size):
        #     src_len = len(alignment_lens[b])
        #     src_mask[b] = float("-inf")
        #     # import ipdb; ipdb.set_trace();
        #     src_mask[b, :src_len, :src_len] = 0.0
        # # src_mask = src_mask.unsqueeze(1)
        # # src_mask = src_mask.repeat(1, NUM_HEADS, 1, 1) 
        # src_mask = src_mask.unsqueeze(0)
        # src_mask = src_mask.repeat(NUM_HEADS, 1, 1, 1) 
        # src_mask = src_mask.view(batch_size*NUM_HEADS, src_seq_len, src_seq_len).contiguous()

    else:
        src_mask = generate_square_subsequent_mask(src_seq_len)
    
    if TGT_MASK_TYPE == "independent":
        tgt_mask = torch.zeros((batch_size, tgt_seq_len, tgt_seq_len))
        for b in range(batch_size):
            alignment_len = alignment_lens[b]
            tgt_mask[b] = create_chunked_tgt_mask(tgt_seq_len, alignment_len)
        tgt_mask = tgt_mask.unsqueeze(1)
        tgt_mask = tgt_mask.repeat(1, NUM_HEADS, 1, 1) 
        tgt_mask = tgt_mask.view(batch_size*NUM_HEADS, tgt_seq_len, tgt_seq_len).contiguous()
        raise NotImplementedError()
    else: # "causal mask"
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    
    
    # import ipdb; ipdb.set_trace();
    memory_mask = torch.zeros((batch_size, tgt_seq_len, src_seq_len))
    if MEMORY_MASK_TYPE == "none":
        pass
    else: # "independent" or "causal"
        for b in range(batch_size):
            alignment_len = alignment_lens[b]
            current_tgt_idx = 0
            for i in range(len(alignment_len)):
                tgt_len = alignment_len[i]
                if MEMORY_MASK_TYPE == "independent":
                    memory_mask[b][current_tgt_idx: current_tgt_idx + tgt_len, :] = float('-inf')
                    memory_mask[b][current_tgt_idx: current_tgt_idx + tgt_len, i] = 0.0 # only corresponding text
                else: # causal
                    memory_mask[b][current_tgt_idx: current_tgt_idx + tgt_len, i + 1:] = float('-inf')
                current_tgt_idx += tgt_len

    # 3D attn_mask requires shape of (bsz * num_heads, tgt_len, src_len)
    memory_mask = memory_mask.unsqueeze(1)
    memory_mask = memory_mask.repeat(1, NUM_HEADS, 1, 1) 
    memory_mask = memory_mask.view(batch_size*NUM_HEADS, tgt_seq_len, src_seq_len).contiguous()

    return src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask

# Custom collate function
def collate_fn(batch):
    text_seqs, unit_seqs_input, unit_seqs_y, align_lens = zip(*batch)

    # text_seqs_padded = pad_sequences(text_seqs, PAD_TOKEN)
    # unit_seqs_input_padded = pad_sequences(unit_seqs_input, PAD_TOKEN)
    # unit_seqs_y_padded = pad_sequences(unit_seqs_y, PAD_TOKEN)

    text_seqs = [torch.Tensor(text_seq).long() for text_seq in text_seqs]
    unit_seqs_input = [torch.Tensor(unit_seq_input).long() for unit_seq_input in unit_seqs_input]
    unit_seqs_y = [torch.Tensor(unit_seq_y).long() for unit_seq_y in unit_seqs_y]
    
    text_seqs_padded = nn.utils.rnn.pad_sequence(text_seqs, batch_first=True, padding_value=PAD_TOKEN)
    unit_seqs_input_padded = nn.utils.rnn.pad_sequence(unit_seqs_input, batch_first=True, padding_value=PAD_TOKEN)
    unit_seqs_y_padded = nn.utils.rnn.pad_sequence(unit_seqs_y, batch_first=True, padding_value=PAD_TOKEN)

    return text_seqs_padded, unit_seqs_input_padded, unit_seqs_y_padded, align_lens

# Training loop
def train(model, dataloader, val_dataloader, optimizer, criterion, device, rank):
    model.train()
    print("Start Training...")
    total_batches = len(dataloader.dataset) // (dataloader.batch_size * dist.get_world_size())
    for epoch in range(EPOCHS):
        for batch_idx, (src, tgt_input, tgt_y, alignment_len) in enumerate(dataloader):
            src, tgt_input, tgt_y = src.to(rank), tgt_input.to(rank), tgt_y.to(rank)

            # debug
            for b in range(src.shape[0]):
                src[b] = src[0]

            src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = create_mask(src, tgt_input, alignment_len)
            src_mask, tgt_mask, memory_mask = src_mask.to(rank), tgt_mask.to(rank), memory_mask.to(rank)
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(rank), tgt_key_padding_mask.to(rank), memory_key_padding_mask.to(rank)

            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
            
            # import ipdb; ipdb.set_trace();
            loss = criterion(output.flatten(end_dim=-2), tgt_y.flatten())
            loss.backward()
            optimizer.step()

            if rank == 0 and WANDB:
                wandb.log({"train_loss": loss.item(), "epoch": epoch + batch_idx / total_batches})

            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(alignment_len)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

            if batch_idx % 250 == 0:
                model.eval()
                val_loss = 0
                num_batches = 0
                with torch.no_grad():
                    for val_src, val_tgt_input, val_tgt_y, val_alignment_len in val_dataloader:
                        val_src, val_tgt_input, val_tgt_y = val_src.to(rank), val_tgt_input.to(rank), val_tgt_y.to(rank)

                        val_src_mask, val_tgt_mask, val_memory_mask, val_src_key_padding_mask, val_tgt_key_padding_mask, val_memory_key_padding_mask = create_mask(val_src, val_tgt_input, val_alignment_len)
                        val_src_mask, val_tgt_mask, val_memory_mask = val_src_mask.to(rank), val_tgt_mask.to(rank), val_memory_mask.to(rank)
                        val_src_key_padding_mask, val_tgt_key_padding_mask, val_memory_key_padding_mask = val_src_key_padding_mask.to(rank), val_tgt_key_padding_mask.to(rank), val_memory_key_padding_mask.to(rank)
                        val_output = model(val_src, val_tgt_input, val_src_mask, val_tgt_mask, val_memory_mask, val_src_key_padding_mask, val_tgt_key_padding_mask, val_memory_key_padding_mask)
                        
                        val_loss += criterion(val_output.flatten(end_dim=-2), val_tgt_y.flatten())
                        num_batches += 1
                val_loss /= num_batches
                print(f'Validation Loss after {batch_idx} batches: {val_loss:.6f}')
                
                if wandb.run:
                    wandb.log({"val_loss": val_loss, "epoch": epoch + batch_idx / total_batches})

                model.train()  # Switch back to training mode
        
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        if rank == 0:
            print("Saving...")
            SAVE_PATH = f"/sailhome/duyy/data/checkpoints/TTU/ckpt-{run_name}/"
            SAVE_NAME = f"epoch-{epoch}.bin"
            os.makedirs(SAVE_PATH, exist_ok = True) 
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, SAVE_NAME))

# Main
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()
    print(f"Running DDP training on rank {rank}.")

    GPU_COUNT = torch.cuda.device_count()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    global run_name
    src_mask_type = f"-{SRC_MASK_TYPE}_src_mask" if SRC_MASK_TYPE != "causal" else ""
    tgt_mask_type = f"-{TGT_MASK_TYPE}_tgt_mask"
    memory_mask_type = f"-{MEMORY_MASK_TYPE}_mem_mask"
    run_name = f"{current_time}-Seperated-NGPU-{GPU_COUNT}_BS-{BATCH_SIZE}_LR-{LR}-EPOCH-{EPOCHS}-EMBED-{EMBED_DIM}-FFN-{FFN_DIM}-DR-{DROPOUT}-NH-{NUM_HEADS}-NL-{NUM_LAYERS}{src_mask_type}{tgt_mask_type}{memory_mask_type}"
    
    if rank == 0 and WANDB:
        # Initialize wandb
        wandb.init(
            project="text-to-unit", 
            name=run_name,
            config={
                "learning_rate": LR,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "embed_dim": EMBED_DIM,
                "num_heads": NUM_HEADS,
                "num_layers": NUM_LAYERS,
                "ffn_dim": FFN_DIM,
                "dropout": DROPOUT,
            }
        )

    # Load real data
    print("Loading data...")
    text_tokens, unit_tokens, alignment = load_data(range(24))  # Example chunks
    print("Loaded.")
    train_dataset = CustomDataset(text_tokens, unit_tokens, alignment)
    train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)

    val_text_tokens, val_unit_tokens, val_alignment = load_data([24])  # Example chunks
    val_text_tokens = val_text_tokens[:1280]
    val_unit_tokens = val_unit_tokens[:1280]
    val_alignment = val_alignment[:1280]

    val_dataset = CustomDataset(val_text_tokens, val_unit_tokens, val_alignment)
    val_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

    model = Seq2SeqTransformer(num_text_tokens=NUM_TEXT_TOKENS, num_unit_tokens=NUM_UNIT_TOKENS, emb_size=EMBED_DIM, nhead=NUM_HEADS,
                               num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    train(model, train_dataloader, val_dataloader, optimizer, criterion, device, rank)

    if rank == 0 and WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()