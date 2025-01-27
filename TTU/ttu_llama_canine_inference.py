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

from llama import Llama
from transformers import AutoConfig
from canine_embedding import CanineEmbeddings
from transformers import Qwen2AudioForConditionalGeneration, Qwen2TokenizerFast

from llama.tokenizer import Tokenizer
from argparse import Namespace
from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
from seamless_communication.streaming.agents.common import AgentStates
import torchaudio

from argparse import ArgumentParser

# TrainingConfig
LR = 1e-5
BATCH_SIZE = 8 # debug, 32
EPOCHS = 4
WANDB = False

DATASET_NAME = "commonvoice"
assert DATASET_NAME in ["mls", "commonvoice"]

# Data Config
CLEAN_TUPAIR = True
ONLY_FIRST = False
SEPERATED = False
assert not (ONLY_FIRST and SEPERATED)

DOUBLE = False
NO_LAST = False

# ModelConfig
EMBED_DIM = 1280  # Embedding dimension
FFN_DIM = 2048 # Default 2048
DROPOUT = 0.1 # Default 0.1
NUM_HEADS = 8 
NUM_LAYERS = 6
SRC_MASK_TYPE = "causal" # none or causal
TGT_MASK_TYPE = "independent" #"independent" or causal or lookbehind-x
MEMORY_MASK_TYPE = "independent" # independent or causal or lookbehind-x

USE_CANINE = True
CANINE_REDUCE = "sum"

ENCODER_TYPE = "llama3"
assert ENCODER_TYPE in ["llama3", "llama3_embed", "qwen2audio"]

UNIT_EMBED_PATH = "/afs/cs.stanford.edu/u/duyy/.cache/fairseq2/assets/aa367aa51efaa58032e42d54/kmeans_10k.npy" # download from https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy
if UNIT_EMBED_PATH is not None:
    assert EMBED_DIM == 1280

# CONTINUE_TRAINING_PATH = None #"/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-08_22-18-05-LLaMA-UnitEmbed-Seperated-Resample-NGPU-3_BS-64_LR-1e-05-Warmup-3000-EPOCH-6-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-4.bin"
CONTINUE_TRAINING_PATH = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-13_01-40-47-LLaMA-UnitEmbed-CANINE_sum-Seperated-Resample-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-1.bin"

assert SRC_MASK_TYPE == "causal" # Llama
if SRC_MASK_TYPE == "none":
    assert MEMORY_MASK_TYPE == "none"

NUM_TEXT_TOKENS = 128_256 # not used
if "llama3" in ENCODER_TYPE:
    TEXT_PAD_TOKEN = -1
elif "qwen2" in ENCODER_TYPE:
    TEXT_PAD_TOKEN = 151643
else:
    raise NotImplementedError()
NUM_UNIT_TOKENS = 10_000 + 2  # Example token size including <eos> and <sos> tokens
SEP_TOKEN = NUM_UNIT_TOKENS - 1
PAD_TOKEN = NUM_UNIT_TOKENS - 2

LLAMA_BOS_TOKEN = 128000
LLAMA3_EMBEDDING_DIM = 4096
QWEN2AUDIO_EMBEDDING_DIM = 4096

CANINE_EMBED_DIM = 768

# Load data function
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(chunks, is_val=False):
    if type(chunks) == int:
        chunks = [chunks]

    # x: list of chunk indices
    if DATASET_NAME == "mls":
        TUPAIR_FILE_TMPL = "/scr-ssd/duyy/MLS_Full_En_TUPair/train-{}-of-04096.pkl"
        ALIGNMENT_FILE_TMPL = "/scr-ssd/duyy/Alignment/result/chunk_{}.pkl"
        MAX_UNIT_SEQ_LEN = 768
    elif DATASET_NAME == "commonvoice" and "llama3" in ENCODER_TYPE:
        if CLEAN_TUPAIR:
            TUPAIR_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/clean/chunk_{}.pkl"
        else:
            TUPAIR_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/chunk_{}.pkl"
        ALIGNMENT_FILE_TMPL = "/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/result/chunk_{}.pkl"
        MAX_UNIT_SEQ_LEN = 1024
    elif DATASET_NAME == "commonvoice" and "qwen2" in ENCODER_TYPE:
        TUPAIR_FILE_TMPL = "/scr-ssd/duyy/qwen2_commonvoice_TUpair/chunk_{}.pkl"
        ALIGNMENT_FILE_TMPL = "/scr-ssd/duyy/Alignment-qwen/result/chunk_{}.pkl"
        MAX_UNIT_SEQ_LEN = 1024
    
    text_tokens = []
    unit_tokens = []
    alignment = []
    for x in chunks:
        # if x in [0, 4, 15, 23]:
        #     continue # TODO: fix this
        if DATASET_NAME == "mls":
            filename = TUPAIR_FILE_TMPL.format(str(x).zfill(5))
        else:
            filename = TUPAIR_FILE_TMPL.format(x)
        if not os.path.isfile(filename):
            continue

        tupair_data = load_pickle(filename)
        chunk_text_tokens = tupair_data['text']
        chunk_unit_tokens = tupair_data['unit']

        chunk_text_tokens = [t for t, u in zip(chunk_text_tokens, chunk_unit_tokens) if len(u) <= MAX_UNIT_SEQ_LEN]
        chunk_unit_tokens = [u for u in chunk_unit_tokens if len(u) <= MAX_UNIT_SEQ_LEN]

        if not os.path.isfile(ALIGNMENT_FILE_TMPL.format(x)):
            continue
        alignment_data = load_pickle(ALIGNMENT_FILE_TMPL.format(x))
        
        # import ipdb; ipdb.set_trace()

        print(x, len(alignment_data), len(chunk_text_tokens))
        try:
            assert len(alignment_data) == len(chunk_text_tokens)
            assert len(alignment_data) == len(chunk_unit_tokens)
        except AssertionError as e:
            import ipdb; ipdb.set_trace();
    
        text_tokens.extend(chunk_text_tokens)
        unit_tokens.extend(chunk_unit_tokens)
        alignment.extend(alignment_data)
    
    for i in range(len(text_tokens)):
        sq_len = len(text_tokens[i])
        # print(sq_len, len(alignment[i]))
        alignment[i] = alignment[i][:sq_len]

    # only first token
    if ONLY_FIRST:
        for i in range(len(text_tokens)):
            text_tokens[i] = text_tokens[i][:1]
            unit_tokens[i] = unit_tokens[i][:alignment[i][0]]
            alignment[i] = alignment[i][:1]
    elif SEPERATED:
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
        text_tokens, unit_tokens, alignment = new_text_tokens, new_unit_tokens, new_alignment
    
    if DOUBLE and not is_val:
        for i in range(len(text_tokens)-1):
            tar_i = i+1
            text_tokens[i] = text_tokens[i] + text_tokens[tar_i]
            unit_tokens[i] = unit_tokens[i] + unit_tokens[tar_i]
            alignment[i] = alignment[i] + alignment[tar_i]
        
    if NO_LAST:
        text_tokens = [t[:-1] for t in text_tokens]
        # Note: we don't care about unit tokens and alignment during inference time, as they are ground truth anyway.
    
    return text_tokens, unit_tokens, alignment

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

def create_chunked_tgt_mask(tgt_seq_len, alignment_len, lookbehind = 0):
    tgt_mask = torch.zeros((tgt_seq_len, tgt_seq_len), dtype=torch.float32)
    current_idx = 0
    previous_idx = []
    try:
        for chunk_size in alignment_len:
            previous_idx.append(current_idx)
            for i in range(current_idx, current_idx + chunk_size):
                # tgt_mask[i, current_idx: i + 1] = 0.0  # intra-chunk causal attention
                # tgt_mask[i, 0: i + 1] = 0.0 # debug: causal

                start_lookbehind_idx = previous_idx[-(lookbehind+1)] if len(previous_idx) >= (lookbehind+1) else 0
                # when lookbehind == 0, previous_idx[-(lookbehind+1)] should always be current_idx

                tgt_mask[i, 0: start_lookbehind_idx] = float("-inf")
                tgt_mask[i, i+1: ] = float("-inf")
            current_idx += chunk_size
    except:
        import ipdb; ipdb.set_trace();

    return tgt_mask

def pad_sequences(sequences, pad_token):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = [seq + [pad_token] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_seqs)

class LlamaDecoder(nn.Module):
    def __init__(self, ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, embed_dim, embed_only, device):
        super(LlamaDecoder, self).__init__()
        
        self.model = Llama.build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size,
                                 model_parallel_size=1) 
        if embed_only:
            self.embed_only = True
            self.model.model = self.model.model.tok_embeddings
        else:
            self.embed_only = False
        self.model.model = self.model.model.to(device).to(torch.bfloat16)

        for k,v in self.model.model.named_parameters():
            v.requires_grad = False

        self.dtype = torch.float32

        feature_dim = LLAMA3_EMBEDDING_DIM # Llama 3

        # Canine Embedding
        if USE_CANINE:
            self.use_canine = True
            self.canine_reduce = CANINE_REDUCE
            assert self.canine_reduce in ["mean", "sum"]
            config = AutoConfig.from_pretrained("google/canine-c")
            embed = CanineEmbeddings(config)
            ckpt_path = "/afs/cs.stanford.edu/u/duyy/data/downloads/canine-c/canine_embedding.bin" # download from https://huggingface.co/google/canine-c/blob/main/pytorch_model.bin
            state_dict = torch.load(ckpt_path)
            embed.load_state_dict(state_dict)

            self.canine_embed = embed
            for k, v in self.canine_embed.named_parameters():
                v.requires_grad = False

            feature_dim += CANINE_EMBED_DIM # Canine-c
        else:
            self.use_canine = False
        
        self.proj = nn.Linear(feature_dim, embed_dim, dtype=self.dtype)
            
    
    def forward(self, src, **kwargs):
        if self.embed_only:
            h = self.model.model(src)
        else:
            h = self.model.model.last_hidden_state(src, 0)

        if self.use_canine:
            tokenizer = self.model.tokenizer
            bs, sq = src.shape
            src_flatten = src.flatten()

            char_ids = []
            char_lens = []
            for i in range(bs*sq):
                token_id = src_flatten[i]
                text = tokenizer.decode([token_id]) if token_id != -1 else ""
                char_id = [ord(c) for c in text]
                char_ids.append(torch.tensor(char_id, dtype=torch.long))
                char_lens.append(len(char_id))
            max_char_lens = max(char_lens)
            char_ids = nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=-1) # (bs*sq_t, sq_c)
            h_canine = self.canine_embed(char_ids) # (bs*sq_t, sq_c, CANINE_EMBED_DIM)
            
            # Create a mask for valid characters
            mask = torch.arange(max_char_lens).expand(len(char_lens), max_char_lens) < torch.tensor(char_lens).unsqueeze(1)
            mask = mask.to(device=h_canine.device)  # Ensure the mask is on the same device as h_canine
            mask = mask.unsqueeze(-1)  # Expand dimensions to match h_canine for broadcasting
            mask = mask.expand(-1, -1, h_canine.size(-1))  # Match the embedding dimension

            # Apply the mask to h_canine
            h_canine_masked = h_canine * mask.float()
            if self.canine_reduce == "mean":
                h_canine_reduced = h_canine_masked.sum(dim=1) / torch.maximum(mask.sum(dim=1).float(), torch.tensor(1).float())
            elif self.canine_reduce == "sum":
                h_canine_reduced = h_canine_masked.sum(dim=1)
            else:
                raise NotImplementedError(f"Canine reduce method {self.canine_reduce} is not defined.")
            # h_canine_reduced shaped (bs*sq_t, CANINE_EMBED_DIM)
            h_canine_reduced = h_canine_reduced.view(bs, sq, CANINE_EMBED_DIM)
            # import ipdb; ipdb.set_trace();
            h = torch.cat([h, h_canine_reduced], dim=-1)

        h = h.to(self.dtype)
        out = self.proj(h)
        return out
class Qwen2Audio(nn.Module):
    def __init__(self, ckpt_dir, embed_dim, embed_only, device):
        super(Qwen2Audio, self).__init__()
        
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(ckpt_dir, device_map="auto")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(ckpt_dir)
        if embed_only:
            raise NotImplementedError("Qwen2-VL embed only not implemented.")
            self.embed_only = True
            self.model.model = self.model.model.tok_embeddings
        else:
            self.embed_only = False
        self.model = self.model.to(device)
        self.device = device

        for k,v in self.model.named_parameters():
            v.requires_grad = False

        self.dtype = torch.float32

        feature_dim = QWEN2AUDIO_EMBEDDING_DIM # Qwen2-VL

        # Canine Embedding
        if USE_CANINE:
            self.use_canine = True
            self.canine_reduce = CANINE_REDUCE
            assert self.canine_reduce in ["mean", "sum"]
            config = AutoConfig.from_pretrained("google/canine-c")
            embed = CanineEmbeddings(config)
            ckpt_path = "/afs/cs.stanford.edu/u/duyy/data/downloads/canine-c/canine_embedding.bin" # download from https://huggingface.co/google/canine-c/blob/main/pytorch_model.bin
            state_dict = torch.load(ckpt_path)
            embed.load_state_dict(state_dict)

            self.canine_embed = embed
            for k, v in self.canine_embed.named_parameters():
                v.requires_grad = False

            feature_dim += CANINE_EMBED_DIM # Canine-c
        else:
            self.use_canine = False
        
        self.proj = nn.Linear(feature_dim, embed_dim, dtype=self.dtype)
            
    
    def forward(self, src, **kwargs):
        if self.embed_only:
            raise NotImplementedError("Qwen2-VL embed only not implemented.")
            h = self.model.model(src)
        else:
            if len(src.shape) == 1:
                src = src.unsqueeze(0)
            assert len(src.shape) == 2, "src.shape should be [bs, sq]"
            attention_mask = torch.ones(src.shape, device=self.device) # we will use right padding, so this should be fine
            # try:
            outputs = self.model(
                input_ids=src,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            h = outputs.hidden_states[-1] # (bs, sq, dim)
            # except RuntimeError as e:
            #     import ipdb; ipdb.set_trace();
        if self.use_canine:
            tokenizer = self.tokenizer
            bs, sq = src.shape
            src_flatten = src.flatten()

            char_ids = []
            char_lens = []
            for i in range(bs*sq):
                token_id = src_flatten[i]
                text = tokenizer.decode([token_id]) if token_id != -1 else ""
                char_id = [ord(c) for c in text]
                char_ids.append(torch.tensor(char_id, dtype=torch.long))
                char_lens.append(len(char_id))
            max_char_lens = max(char_lens)
            char_ids = nn.utils.rnn.pad_sequence(char_ids, batch_first=True, padding_value=-1) # (bs*sq_t, sq_c)
            char_ids = char_ids.to(self.device)
            h_canine = self.canine_embed(char_ids) # (bs*sq_t, sq_c, CANINE_EMBED_DIM)
            
            # Create a mask for valid characters
            mask = torch.arange(max_char_lens).expand(len(char_lens), max_char_lens) < torch.tensor(char_lens).unsqueeze(1)
            mask = mask.to(device=h_canine.device)  # Ensure the mask is on the same device as h_canine
            mask = mask.unsqueeze(-1)  # Expand dimensions to match h_canine for broadcasting
            mask = mask.expand(-1, -1, h_canine.size(-1))  # Match the embedding dimension

            # Apply the mask to h_canine
            h_canine_masked = h_canine * mask.float()
            if self.canine_reduce == "mean":
                h_canine_reduced = h_canine_masked.sum(dim=1) / torch.maximum(mask.sum(dim=1).float(), torch.tensor(1).float())
            elif self.canine_reduce == "sum":
                h_canine_reduced = h_canine_masked.sum(dim=1)
            else:
                raise NotImplementedError(f"Canine reduce method {self.canine_reduce} is not defined.")
            # h_canine_reduced shaped (bs*sq_t, CANINE_EMBED_DIM)
            h_canine_reduced = h_canine_reduced.view(bs, sq, CANINE_EMBED_DIM)
            # import ipdb; ipdb.set_trace();
            h = torch.cat([h, h_canine_reduced], dim=-1)

        h = h.to(self.dtype)
        out = self.proj(h)
        return out
    
# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_text_tokens, num_unit_tokens, emb_size, nhead, custom_encoder, num_decoder_layers):
        super(Seq2SeqTransformer, self).__init__()
        self.text_encoder = custom_encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead, 
                                          dim_feedforward=FFN_DIM, dropout=DROPOUT,
                                          batch_first=True,)
        decoder_norm = nn.LayerNorm(emb_size)
        self.unit_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.unit_embedding = nn.Embedding(num_unit_tokens, emb_size)
        self.fc_out = nn.Linear(emb_size, num_unit_tokens)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # import ipdb; ipdb.set_trace();
        memory = self.text_encoder(src)
        tgt_emb = self.unit_embedding(tgt)
        outs = self.unit_decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,)
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
        # raise NotImplementedError()
    elif TGT_MASK_TYPE.startswith("lookbehind"):
        lookbehind = int(TGT_MASK_TYPE.replace("lookbehind-", ""))
        tgt_mask = torch.zeros((batch_size, tgt_seq_len, tgt_seq_len))
        for b in range(batch_size):
            alignment_len = alignment_lens[b]
            tgt_mask[b] = create_chunked_tgt_mask(tgt_seq_len, alignment_len, lookbehind)
        tgt_mask = tgt_mask.unsqueeze(1)
        tgt_mask = tgt_mask.repeat(1, NUM_HEADS, 1, 1) 
        tgt_mask = tgt_mask.view(batch_size*NUM_HEADS, tgt_seq_len, tgt_seq_len).contiguous()
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
                elif MEMORY_MASK_TYPE.startswith("lookbehind"):
                    lookbehind = int(MEMORY_MASK_TYPE.replace("lookbehind-", ""))
                    memory_mask[b][current_tgt_idx: current_tgt_idx + tgt_len, :] = float('-inf')
                    memory_mask[b][current_tgt_idx: current_tgt_idx + tgt_len, max(0, i-lookbehind) : i+1] = 0.0 # only corresponding text and lookbehind x
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
    text_seqs = [torch.tensor(text_seq, dtype=torch.long) for text_seq in text_seqs]
    unit_seqs_input = [torch.tensor(unit_seq_input, dtype=torch.long) for unit_seq_input in unit_seqs_input]
    unit_seqs_y = [torch.tensor(unit_seq_y, dtype=torch.long) for unit_seq_y in unit_seqs_y]

    text_seqs_padded = nn.utils.rnn.pad_sequence(text_seqs, batch_first=True, padding_value=TEXT_PAD_TOKEN) # -1 for llama tokenizer <PAD>
    unit_seqs_input_padded = nn.utils.rnn.pad_sequence(unit_seqs_input, batch_first=True, padding_value=PAD_TOKEN)
    unit_seqs_y_padded = nn.utils.rnn.pad_sequence(unit_seqs_y, batch_first=True, padding_value=PAD_TOKEN)

    return text_seqs_padded, unit_seqs_input_padded, unit_seqs_y_padded, align_lens

# Training loop
def train(model, dataloader, val_dataloader, optimizer, criterion, device, rank):
    dtype = torch.bfloat16
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
            src_mask, tgt_mask, memory_mask = src_mask.to(dtype).to(rank), tgt_mask.to(dtype).to(rank), memory_mask.to(dtype).to(rank)
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(dtype).to(rank), tgt_key_padding_mask.to(dtype).to(rank), memory_key_padding_mask.to(dtype).to(rank)

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

            if batch_idx % 1000 == 0:
                model.eval()
                val_loss = 0
                num_batches = 0
                with torch.no_grad():
                    for val_src, val_tgt_input, val_tgt_y, val_alignment_len in val_dataloader:
                        val_src, val_tgt_input, val_tgt_y = val_src.to(rank), val_tgt_input.to(rank), val_tgt_y.to(rank)

                        val_src_mask, val_tgt_mask, val_memory_mask, val_src_key_padding_mask, val_tgt_key_padding_mask, val_memory_key_padding_mask = create_mask(val_src, val_tgt_input, val_alignment_len)
                        val_src_mask, val_tgt_mask, val_memory_mask = val_src_mask.to(dtype).to(rank), val_tgt_mask.to(dtype).to(rank), val_memory_mask.to(dtype).to(rank)
                        val_src_key_padding_mask, val_tgt_key_padding_mask, val_memory_key_padding_mask = val_src_key_padding_mask.to(dtype).to(rank), val_tgt_key_padding_mask.to(dtype).to(rank), val_memory_key_padding_mask.to(dtype).to(rank)
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
    run_name = f"{current_time}-LLaMA-NGPU-{GPU_COUNT}_BS-{BATCH_SIZE}_LR-{LR}-EPOCH-{EPOCHS}-EMBED-{EMBED_DIM}-FFN-{FFN_DIM}-DR-{DROPOUT}-NH-{NUM_HEADS}-NL-{NUM_LAYERS}{src_mask_type}{tgt_mask_type}{memory_mask_type}"
    
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
    text_tokens, unit_tokens, alignment = load_data(range(0))  # Example chunks
    
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

    ckpt_dir = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct'
    tokenizer_path = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct/tokenizer.model'
    max_seq_len = 64 # actually 48 for raw text token in parsed commonvoice
    max_batch_size = BATCH_SIZE
    llama_model = LlamaDecoder(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, EMBED_DIM, rank) 
    
    model = Seq2SeqTransformer(num_text_tokens=NUM_TEXT_TOKENS, num_unit_tokens=NUM_UNIT_TOKENS, emb_size=EMBED_DIM, nhead=NUM_HEADS,
                               custom_encoder=llama_model, num_decoder_layers=NUM_LAYERS)
    model = model.to(torch.bfloat16).to(rank)

    # import ipdb; ipdb.set_trace()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DDP(model, device_ids=[rank])

    train(model, train_dataloader, val_dataloader, optimizer, criterion, device, rank)

    if rank == 0 and WANDB:
        wandb.finish()

# beam_search_decode(model, src, src_mask, max_len, start_symbol, sep_token, beam_size=10)
def beam_search_decode(model, src, src_mask, max_len, start_symbol, sep_token, beam_size=10):
    dtype = torch.float32

    src = src.to(model.device)
    src_mask = src_mask.to(model.device)

    # Initialize the beam with the start symbol
    ys = torch.full((beam_size, 1), start_symbol, dtype=torch.long, device=model.device)
    beam_scores = torch.zeros(beam_size, device=model.device)  # Initial scores for beams are 0
    alignment_lens = [[[0]] * beam_size]  # List of lists of lists to track alignment lengths for each beam

    completed_hypotheses = []
    completed_scores = []

    for step in range(max_len):
        all_candidates = []
        all_scores = []
        all_alignment_lens = []

        for i in range(len(ys)):
            if SRC_MASK_TYPE == "none":
                cur_src = src
            else:
                cur_src = src[:, :i+1]
                
            hypothesis = ys[i:i+1]
            hypothesis_score = beam_scores[i]
            hypothesis_alignment_lens = alignment_lens[i]

            # Update alignment lengths for the current step
            for b in range(len(hypothesis_alignment_lens)):
                hypothesis_alignment_lens[b][-1] += 1

            # Prepare masks and model input
            src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = create_mask(cur_src, hypothesis, hypothesis_alignment_lens)
            src_mask, tgt_mask, memory_mask = src_mask.to(dtype), tgt_mask.to(dtype), memory_mask.to(dtype)
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(dtype), tgt_key_padding_mask.to(dtype), memory_key_padding_mask.to(dtype)

            out = model(cur_src, hypothesis, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
            out = out[:, -1]  # Get the last token outputs across all beams

            probs = torch.softmax(out, dim=-1)
            top_probs, top_idx = probs.topk(beam_size)  # Get top beam_size possibilities from current beam

            for j in range(beam_size):
                next_word = top_idx[j]
                score = hypothesis_score + torch.log(top_probs[j])  # Accumulate log probabilities for numerical stability
                candidate = torch.cat([hypothesis, next_word.unsqueeze(0)])
                candidate_alignment_lens = [x[:] for x in hypothesis_alignment_lens]  # Deep copy of alignment lengths

                if next_word == sep_token:
                    candidate_alignment_lens.append([0])

                all_candidates.append(candidate)
                all_scores.append(score)
                all_alignment_lens.append(candidate_alignment_lens)

        # After expanding all beams, sort by scores and select the top beam_size ones
        all_scores = torch.stack(all_scores)
        top_scores, top_candidates_idx = all_scores.topk(beam_size)

        # Update current beams and scores
        ys = torch.stack([all_candidates[idx] for idx in top_candidates_idx])
        beam_scores = top_scores
        alignment_lens = [all_alignment_lens[idx] for idx in top_candidates_idx]

        # Check for completed hypotheses (those with enough sep_tokens)
        to_remove = []
        for i, candidate_alignment_lens in enumerate(alignment_lens):
            if len(candidate_alignment_lens) == len(src[0]) + 1:  # +1 for the start symbol
                completed_hypotheses.append(ys[i])
                completed_scores.append(beam_scores[i])
                to_remove.append(i)

        # Remove completed hypotheses from current beams
        if to_remove:
            ys = torch.stack([ys[i] for i in range(len(ys)) if i not in to_remove])
            beam_scores = torch.stack([beam_scores[i] for i in range(len(beam_scores)) if i not in to_remove])
            alignment_lens = [alignment_lens[i] for i in range(len(alignment_lens)) if i not in to_remove]

        # Break if all beams end or max_len reached
        if len(ys) == 0:
            break

    if not completed_hypotheses:
        # If no complete hypotheses, take whatever we have as the final output
        completed_hypotheses = ys
        completed_scores = beam_scores

    # Select the best one based on the scores
    best_hypothesis_idx = torch.argmax(torch.tensor(completed_scores))
    best_hypothesis = completed_hypotheses[best_hypothesis_idx]

    return best_hypothesis, alignment_lens[best_hypothesis_idx]

def greedy_decode(model, src, src_mask, max_len, start_symbol, sep_token, answer=None, gt_align=None):
    dtype = torch.float32

    src = src.to(model.device)
    src_mask = src_mask.to(model.device)
    
    # memory = model.transformer.encoder(model.text_embedding(src), src_mask)
    ys = torch.ones(1, 1, dtype=torch.long).fill_(start_symbol).type(torch.long).to(model.device)
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
            
            src_mask, tgt_mask, memory_mask = src_mask.to(dtype).to(model.device), tgt_mask.to(dtype).to(model.device), memory_mask.to(dtype).to(model.device)
            src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask = src_key_padding_mask.to(dtype).to(model.device), tgt_key_padding_mask.to(dtype).to(model.device), memory_key_padding_mask.to(dtype).to(model.device)
            
            out = model(cur_src, ys, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
            # if i > 4:
            # if True:
            # if i == 0 and j == 0:
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

    # import ipdb; ipdb.set_trace();
        
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
def demo(tokenizer, vocoder_agent, text_tokens, unit_tokens, model_name, id, output_path=None):
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
    
    # get text
    text = tokenizer.decode(text_tokens)
    print("[TEXT]: ", text)
    # get audio
    states = AgentStates()
    states.source = unit_tokens
    y = vocoder_agent.policy(states)
    try:
        content = torch.tensor(y.content.content, dtype=torch.float32, device='cpu').unsqueeze(0)
        if output_path:
            output_folder = os.path.join(output_path, model_name)
        else:
            output_folder = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/{model_name}"
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"{id}_{text}.wav")
        # import ipdb; ipdb.set_trace()
        torchaudio.save(output_file, content, sample_rate=16000)
    except Exception as e:
        print("no", e)
        pass
def main_inference(args):
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-02_14-03-44-LLaMA-NGPU-1_BS-64_LR-1e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-3.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-03_01-56-41-LLaMA-NGPU-1_BS-64_LR-1e-05-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-03_17-06-37-LLaMA-NGPU-4_BS-64_LR-1e-05-Warmup-2000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-2.bin"
    
    # overfit
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-03_22-24-02-LLaMA-NGPU-4_BS-64_LR-1e-05-Warmup-10-EPOCH-5000-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-4999.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-03_23-41-35-LLaMA-NGPU-1_BS-8_LR-1e-05-Warmup-10-EPOCH-5000-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-4999.bin"

    # only first
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-04_15-37-46-LLaMA-OnlyFirst-NGPU-2_BS-64_LR-1e-05-Warmup-4000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-04_16-10-36-LLaMA-OnlyFirst-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-3.bin"
    
    # (one token) with resample
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-04_22-24-15-LLaMA-OnlyFirst-Resample-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-3.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-05_00-25-31-LLaMA-Seperated-Resample-NGPU-4_BS-64_LR-1e-05-Warmup-6000-EPOCH-5-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-1.bin"

    # independent memory mask
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-05_10-29-19-LLaMA-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-5-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-3.bin"

    # weight_sep (independent mem mask)
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-05_16-06-51-LLaMA-WEIGHT_SEP-5-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-2.bin"

    # resample
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-05_16-56-12-LLaMA-Resample-WEIGHT_SEP-5-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-0.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-05_23-57-24-LLaMA-Resample-NGPU-4_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-0.bin"

    # 08-07
    # causal causal
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-07_01-13-24-LLaMA-NGPU-4_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-3.bin"

    # seperated (independent-independent)
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-07_02-43-31-LLaMA-Seperated-NGPU-2_BS-64_LR-1e-05-Warmup-6000-EPOCH-4-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-2.bin"

    # UnitEmbed causal causal
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-07_10-08-23-LLaMA-UnitEmbed-NGPU-4_BS-32_LR-1e-05-Warmup-3000-EPOCH-4-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-3.bin"

    # causal ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-07_10-09-11-LLaMA-NGPU-2_BS-64_LR-1e-05-Warmup-3000-EPOCH-4-EMBED-1024-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-1.bin"

    # Continue UnitEmbed ind ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-10_18-24-17-LLaMA-Continue-NGPU-4_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"

    # Continue UnitEmbed causal causal
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-11_17-19-59-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-causal_mem_mask/epoch-4.bin"

    # Continue UnitEmbed lookbehind-1 lookbehind-1
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-12_00-17-01-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-1_tgt_mask-lookbehind-1_mem_mask/epoch-4.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-12_00-20-39-LLaMA-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-2_tgt_mask-lookbehind-2_mem_mask/epoch-4.bin"
    
    # Continue UnitEmbed causal ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-14_13-55-57-LLaMA-Continue-NGPU-4_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-causal_tgt_mask-independent_mem_mask/epoch-4.bin"

    # CANINE sum ind ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-15_11-18-01-LLaMA-UnitEmbed-CANINE_sum-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"

    # llama3_embed CANINE sum ind ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-20_20-56-36-LLaMA-UnitEmbed-CANINE_sum-Continue-NGPU-2_BS-64_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-3.bin"

    # llama3_embed ind ind
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-20_20-58-32-LLaMA-UnitEmbed-Continue-NGPU-2_BS-64_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"

    # CANINE llama3 ind ind (double)
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-09-29_06-57-10-llama3-UnitEmbed-CANINE_sum-Continue-Double-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-3-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-2.bin"
    # direct double
    # model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-03_00-29-52-llama3-UnitEmbed-CANINE_sum-Continue-Double-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-3.bin"

    # qwen2audio canine ind ind
    # model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-09_03-27-25-qwen2audio-UnitEmbed-CANINE_sum-Continue-NGPU-1_BS-20_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"

    # CANINE llama3 ind ind (alignment-ctc)
    # model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-12_11-00-27-llama3-UnitEmbed-CANINE_sum-CTCAlign-Resample-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-1.bin"
    # model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-14_08-32-57-llama3-UnitEmbed-CANINE_sum-Continue-CTCAlign-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"
    # merged alignment
    # model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-16_03-10-09-llama3-UnitEmbed-CANINE_sum-Continue-CTCAlign-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"
    # lookbenind-1 lookbenind-1
    model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-10-17_06-25-06-llama3-UnitEmbed-CANINE_sum-Continue-CTCAlign-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-lookbehind-1_tgt_mask-lookbehind-1_mem_mask/epoch-4.bin"

    # Override is there's path in args
    if args.model_path:
        model_path = args.model_path

    model_name = os.path.basename(os.path.dirname(model_path))
    ckpt_name = os.path.basename(model_path).replace(".bin", "")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()

    # ckpt_dir = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct'
    # tokenizer_path = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct/tokenizer.model'
    ckpt_dir = '/scr/biggest/duyy/Meta-Llama-3-8B-Instruct/'
    tokenizer_path = '/scr/biggest/duyy/Meta-Llama-3-8B-Instruct/tokenizer.model'
    qwen_ckpt_dir = '/scr/biggest/duyy/Qwen2-Audio-7B-Instruct'
    max_seq_len = 64 # actually 48 for raw text token in parsed commonvoice
    max_batch_size = BATCH_SIZE
    if ENCODER_TYPE == "llama3":
        encoder = LlamaDecoder(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, EMBED_DIM, False, rank) 
    elif ENCODER_TYPE == "llama3_embed":
        encoder = LlamaDecoder(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, EMBED_DIM, True, rank) 
    elif ENCODER_TYPE == "qwen2audio":
        encoder = Qwen2Audio(qwen_ckpt_dir, EMBED_DIM, False, rank)

    model = Seq2SeqTransformer(num_text_tokens=NUM_TEXT_TOKENS, num_unit_tokens=NUM_UNIT_TOKENS, emb_size=EMBED_DIM, nhead=NUM_HEADS,
                               custom_encoder=encoder, num_decoder_layers=NUM_LAYERS)
    model.load_state_dict(parse_state_dict(torch.load(model_path)), strict=False)
    model = model.to(torch.float32).to(rank)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.device = device


    text_tokens, unit_tokens, alignment = load_data([24]) # 24 or 4095
    print("data done.")
    
    # src_sentence = "Example source sentence"  # This should be tokenized
    # src = torch.tensor([src_sentence])  # Replace this with actual tokenized input
    
    LLAMA3_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    if ENCODER_TYPE in ["llama3", "llama3_embed"]:
        tokenizer = Tokenizer(LLAMA3_TOKENIZER_URI)
    elif ENCODER_TYPE == "qwen2audio":
        tokenizer = encoder.tokenizer

    
    vocoder_args = Namespace(
        vocoder_name='vocoder_v2',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32,
        sample_rate=16000,
        tgt_lang='eng',
        vocoder_speaker_id=-1,
    )
    vocoder_agent = VocoderAgent(vocoder_args)

    ss = ["I love my cat.",
          "An apple a day.",
          "Each day he walks.",
          "Cold nights make the stars.",
          "We watch the waves.",
          "She wears her red dress",
          "They hear the loud calls."]
    for i in range(50, 100):
    # for i in range(len(ss)):
        src = torch.tensor(text_tokens[i:i+1], dtype=torch.long)
        answer = torch.tensor(unit_tokens[i:i+1], dtype=torch.long).to('cuda')
        gt_align = torch.tensor(alignment[i:i+1], dtype=torch.long)

        # ablation
        
        # from llama.tokenizer import Tokenizer
        # DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
        # tokenizer = Tokenizer(DEFAULT_TOKENIZER_URI)
        # s = ss[i]
        # src = tokenizer.encode(s, bos=False, eos=False)
        # src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
        # import ipdb; ipdb.set_trace();


        src_mask = generate_square_subsequent_mask(src.size(1))
        
        start_symbol = SEP_TOKEN # We use sep token as start symbol
        sep_token = SEP_TOKEN
        
        # import ipdb; ipdb.set_trace();
        # beam_search_decode(model, src, src_mask, max_len, start_symbol, sep_token, beam_size=10)
        print("Start greedy decode")
        predicted_tokens = greedy_decode(model, src, src_mask, max_len=1000, start_symbol=start_symbol, sep_token=sep_token, answer=answer, gt_align=gt_align)
        print("End greedy decode")

        # predicted_tokens = answer
        if not args.output_path:
            demo(tokenizer, vocoder_agent, src.squeeze(0), predicted_tokens, model_name, i, output_path=args.output_path)
        else:
            demo(tokenizer, vocoder_agent, src.squeeze(0), predicted_tokens, ckpt_name, i, output_path=args.output_path)
        print("Predicted Tokens: ", predicted_tokens)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    args = parser.parse_args()
    print(args)

    main_inference(args)