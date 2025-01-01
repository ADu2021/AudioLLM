import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

import os
import pickle
import numpy
from tqdm import tqdm
import lzma

import multiprocessing
import os
from argparse import ArgumentParser
import datetime
import subprocess

from llama import Llama

BATCH_SIZE=128

def now():
    return datetime.datetime.now()

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

    max_text_token_lens = 0
    for text_token in text_tokens:
        max_text_token_lens = max(max_text_token_lens, len(text_token))
    print("max_text_token_lens:", max_text_token_lens) # max_text_token_lens: 48

    return text_tokens, unit_tokens, alignment


class TextDataset(Dataset):
    def __init__(self, text_tokens):
        self.text_tokens = text_tokens

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        return self.text_tokens[idx]

def collate_fn(batch):
    """ Collate function to pad the sequences in the batch. """
    text_tokens = batch

    text_tokens_padded = nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=-1)

    return text_tokens_padded


def get_log_file(x):
    return f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/result/logs/chunk_{x}.log"
def job(gpu_id: int, x: int, model):
    print(f"chunk {x} @ {gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job started at {now()}\n")
    
    text_tokens, _, _ = load_data([x])
    text_tokens = [torch.Tensor(t).long() for t in text_tokens]
    dataset = TextDataset(text_tokens)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    

    results = []

    for (text_tokens) in tqdm(train_loader):
        text_tokens = text_tokens.to('cuda')

        output = model.model.last_hidden_state(text_tokens, 0)

        results.append(output)

    output_file = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/TTU/result/chunk_{x}.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job finished at {now()}\n")

if __name__ == "__main__":
    # ckpt_dir = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct'
    # tokenizer_path = '/afs/cs.stanford.edu/u/duyy/data/checkpoints/Meta-Llama-3-8B-Instruct/tokenizer.model'
    # max_seq_len = 256
    # max_batch_size = BATCH_SIZE
    # model = Llama.build(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size) 
    # model.model = model.model.to('cuda')
    model = None
    # for i in range(25):
        # job(0, i, model)
    load_data(range(25))