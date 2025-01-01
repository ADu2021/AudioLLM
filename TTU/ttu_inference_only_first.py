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
BATCH_SIZE = 64
EPOCHS = 4
WANDB = False

# ModelConfig - Inference
EMBED_DIM = 1024  # Embedding dimension
FFN_DIM = 2048 # Default 2048
DROPOUT = 0.1 # Default 0.1
NUM_HEADS = 8 
NUM_LAYERS = 6
SRC_MASK_TYPE = "none" # none or causal
TGT_MASK_TYPE = "causal" #"independent" or causal
MEMORY_MASK_TYPE = "none" # none, independent or causal

NUM_TEXT_TOKENS = 128_256
NUM_UNIT_TOKENS = 10_000 + 2  # Example token size including <eos> and <sos> tokens
# SOS_TOKEN = NUM_UNIT_TOKENS - 3
# EOS_TOKEN = NUM_UNIT_TOKENS - 4
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

    # # only first token
    # for i in range(len(text_tokens)):
    #     text_tokens[i] = text_tokens[i][:1]
    #     unit_tokens[i] = unit_tokens[i][:alignment[i][0]]
    #     alignment[i] = alignment[i][:1]

    # return text_tokens, unit_tokens, alignment

    # seperated
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

        aligned_unit_seq_input = []
        aligned_unit_seq_y = []
        cur_pos = 0
        for i, length in enumerate(align_len):
            aligned_unit_seq_input.append(SEP_TOKEN)  # Note that we use SOS here
            aligned_unit_seq_input.extend(unit_seq[cur_pos: cur_pos + length])

            aligned_unit_seq_y.extend(unit_seq[cur_pos: cur_pos + length])
            aligned_unit_seq_y.append(SEP_TOKEN)  # and use EOS here

            cur_pos += length
            align_len[i] += 1  # since we append SOS token for each chunk

        return text_seq, aligned_unit_seq_input, aligned_unit_seq_y, align_len

def create_chunked_tgt_mask(tgt_seq_len, alignment_len):
    tgt_mask = torch.full((tgt_seq_len, tgt_seq_len), float('-inf'))
    current_idx = 0

    for chunk_size in alignment_len:
        for i in range(current_idx, current_idx + chunk_size):
            # tgt_mask[i, current_idx: i + 1] = 0.0  # Causal and intra-chunk attention
            tgt_mask[i, 0: i + 1] = 0.0
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

            src_mask, tgt_mask, memory_mask = create_mask(src, tgt_input, alignment_len)
            src_mask, tgt_mask, memory_mask = src_mask.to(rank), tgt_mask.to(rank), memory_mask.to(rank)
            # import ipdb; ipdb.set_trace();
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask, memory_mask)
            
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

                        val_src_mask, val_tgt_mask, val_memory_mask = create_mask(val_src, val_tgt_input, val_alignment_len)
                        val_src_mask, val_tgt_mask, val_memory_mask = val_src_mask.to(rank), val_tgt_mask.to(rank), val_memory_mask.to(rank)
                        val_output = model(val_src, val_tgt_input, val_src_mask, val_tgt_mask, val_memory_mask)
                        
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
    tgt_mask_type = f"-{TGT_MASK_TYPE}_tgt_mask"
    run_name = f"{current_time}-NGPU-{GPU_COUNT}_BS-{BATCH_SIZE}_LR-{LR}-EPOCH-{EPOCHS}-EMBED-{EMBED_DIM}-FFN-{FFN_DIM}-DR-{DROPOUT}-NH-{NUM_HEADS}-NL-{NUM_LAYERS}-{tgt_mask_type}"
    
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
    text_tokens, unit_tokens, alignment = load_data(range(24))  # Example chunks
    
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

def demo_sep(text_tokens, unit_tokens, align, id):
    from llama.tokenizer import Tokenizer
    from argparse import Namespace
    from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
    from seamless_communication.streaming.agents.common import AgentStates
    import torchaudio

    try:
        text_tokens = text_tokens.tolist()
    except:
        pass
    try:
        align = align.tolist()
    except:
        pass

    non_sep_mask = unit_tokens != SEP_TOKEN
    unit_tokens = unit_tokens[non_sep_mask]
    
    unit_tokens = unit_tokens.unsqueeze(0).unsqueeze(0)
    

    DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    tokenizer = Tokenizer(DEFAULT_TOKENIZER_URI)
    # get text
    text = tokenizer.decode(text_tokens)
    print("[TEXT]: ", text)
    # get audio
    args = Namespace(
        vocoder_name='vocoder_v2',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32,
        sample_rate=16000,
        tgt_lang='eng',
        vocoder_speaker_id=-1,
    )
    vocoder_agent = VocoderAgent(args)
    start_pos = 0
    for i in range(len(text_tokens)):
        end_pos = start_pos + align[i]
        text_token = tokenizer.decode(text_tokens[i:i+1])
        states = AgentStates()
        states.source = unit_tokens[..., start_pos : end_pos]
        # import ipdb; ipdb.set_trace()
        y = vocoder_agent.policy(states)
        try:
            content = torch.Tensor(y.content.content).unsqueeze(0)
            output_folder = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/ademo/demo-{id}"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{i}_{text_tokens[i]}_{text_token}.wav")
            # import ipdb; ipdb.set_trace()
            torchaudio.save(output_path, content, sample_rate=args.sample_rate)
        except:
            print("no")
            pass
        start_pos = end_pos
    
def greedy_decode(model, src, src_mask, max_len, start_symbol, sep_token, answer):
    
    src = src.to(model.device)
    src_mask = src_mask.to(model.device)
    
    # memory = model.transformer.encoder(model.text_embedding(src), src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    alignment_lens = [[0] * src.shape[0]]

    for i in range(len(src[0])):
        # import ipdb; ipdb.set_trace();
        if SRC_MASK_TYPE == "none":
            cur_src = src
        else:
            cur_src = src[:, :i+1]
        # cur_src_mask = src_mask[:i+1, :i+1]
        # cur_memory = model.transformer.encoder(model.text_embedding(cur_src), cur_src_mask)  # TODO: use kv-cache here.
        for j in range(max_len-1):
            for b in range(len(alignment_lens)):
                alignment_lens[b][-1] += 1

            src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = create_mask(cur_src, ys, alignment_lens)
            
            src_mask, tgt_mask, memory_mask = src_mask.to(model.device), tgt_mask.to(model.device), memory_mask.to(model.device)
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(model.device), tgt_key_padding_mask.to(model.device), memory_key_padding_mask.to(model.device)
            
            out = model(cur_src, ys, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
            # if j == 0:
            # # if True:
            #     import ipdb; ipdb.set_trace();
            out = out[:, -1]
            _, next_word = torch.max(out, dim=1)
            next_word = next_word.item()
            
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            if next_word == sep_token:
                break
        
        for b in range(len(alignment_lens)):
            alignment_lens[b].append(0)
    for b in range(len(alignment_lens)):
        alignment_lens[b][-1] += 1 # for the last sep token
        
    # alignment_lens = alignment_lens[0][:-1]
    # demo_sep(src.squeeze(), ys, alignment_lens, 0)
    # import ipdb; ipdb.set_trace();

    return ys
def parse_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict
# Example usage
def demo(text_tokens, unit_tokens, model_name, id):
    from llama.tokenizer import Tokenizer
    from argparse import Namespace
    from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
    from seamless_communication.streaming.agents.common import AgentStates
    import torchaudio

    try:
        text_tokens = text_tokens.tolist()
    except:
        pass
    try:
        align = align.tolist()
    except:
        pass
    
    non_sep_mask = (unit_tokens < PAD_TOKEN)
    unit_tokens = unit_tokens[non_sep_mask]

    unit_tokens = unit_tokens.unsqueeze(0).unsqueeze(0)
    


    DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    tokenizer = Tokenizer(DEFAULT_TOKENIZER_URI)
    # get text
    text = tokenizer.decode(text_tokens)
    print("[TEXT]: ", text)
    # get audio
    args = Namespace(
        vocoder_name='vocoder_v2',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32,
        sample_rate=16000,
        tgt_lang='eng',
        vocoder_speaker_id=-1,
    )
    vocoder_agent = VocoderAgent(args)
    states = AgentStates()
    states.source = unit_tokens
    y = vocoder_agent.policy(states)
    try:
        content = torch.Tensor(y.content.content).unsqueeze(0)
        output_folder = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/{model_name}/"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"{id}_{text}.wav")
        # import ipdb; ipdb.set_trace()
        torchaudio.save(output_path, content, sample_rate=args.sample_rate)
    except:
        print("no")
        pass

def main_inference():
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-29_20-08-14-NGPU-2_BS-64_LR-0.0001-EPOCH-4-causal_tgt_mask/epoch-3.bin"
    model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-29_21-03-24-NGPU-2_BS-32_LR-3e-05-EPOCH-4-EMBED-1024-NH-8-NL-6--causal_tgt_mask/epoch-3.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-30_12-46-06-NGPU-2_BS-32_LR-3e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.5-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-30_22-12-08-NGPU-2_BS-32_LR-3e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-none_src_mask-causal_tgt_mask-none_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-31_15-17-56-NGPU-2_BS-32_LR-3e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-none_src_mask-causal_tgt_mask-none_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-31_20-14-17-NGPU-2_BS-64_LR-1e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-0.bin"

    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-31_23-53-21-Seperated-NGPU-2_BS-32_LR-5e-06-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-none_src_mask-causal_tgt_mask-none_mem_mask/epoch-1.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-07-31_23-05-13-OnlyFirst-NGPU-2_BS-32_LR-5e-06-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-none_src_mask-causal_tgt_mask-none_mem_mask/epoch-0.bin"

    model_name = os.path.basename(os.path.dirname(model_path))
    model = Seq2SeqTransformer(num_text_tokens=NUM_TEXT_TOKENS, num_unit_tokens=NUM_UNIT_TOKENS, emb_size=EMBED_DIM, nhead=NUM_HEADS,
                               num_encoder_layers=NUM_LAYERS, num_decoder_layers=NUM_LAYERS)
    model.load_state_dict(parse_state_dict(torch.load(model_path)))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.device = device


    text_tokens, unit_tokens, alignment = load_data([24])
    
    # src_sentence = "Example source sentence"  # This should be tokenized
    # src = torch.tensor([src_sentence])  # Replace this with actual tokenized input
    for i in range(10):
        src = torch.Tensor(text_tokens[i:i+1]).long()
        answer = torch.Tensor(unit_tokens[i:i+1]).long().to('cuda')

        src_mask = generate_square_subsequent_mask(src.size(1))
        
        start_symbol = SEP_TOKEN # We use sep token as start symbol
        sep_token = SEP_TOKEN
        
        # import ipdb; ipdb.set_trace();
        predicted_tokens = greedy_decode(model, src, src_mask, max_len=1000, start_symbol=start_symbol, sep_token=sep_token, answer=answer)
        

        demo(src.squeeze(0), predicted_tokens, model_name, i)
        print("Predicted Tokens: ", predicted_tokens)

if __name__ == "__main__":
    main_inference()