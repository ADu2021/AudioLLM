## Convert from a pair of (text, audio) to a pair of (text tokens, unit tokens)
## Use conda env seamless @ /nlp/scr/duyy/miniconda3/envs/seamless/bin/python

from seamless_communication.inference import Translator
from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
from seamless_communication.streaming.agents.common import AgentStates
from fairseq2.typing import Device
import torch
from llama.tokenizer import Tokenizer
from transformers import AutoTokenizer

import numpy
import pickle
from tqdm import tqdm

import multiprocessing
import os
from argparse import ArgumentParser
import datetime
import subprocess
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from argparse import Namespace
import time

from ctc_forced_alignment import ForcedAligner, save_token_segments
from utils import scale_to_sum

TEXT_PADDING_VALUE = 128_256
UNIT_PADDING_VALUE = 10_000

def now():
    return datetime.datetime.now()

DEFAULT_KMEANS_URI = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"

class TextUnitDataset(Dataset):
    """ Dataset class to handle text and unit token pairs. """
    def __init__(self, text_tokens, unit_tokens):
        self.text_tokens = text_tokens
        self.unit_tokens = unit_tokens

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        return self.text_tokens[idx], self.unit_tokens[idx]

def collate_fn(batch):
    """ Collate function to pad the sequences in the batch. """
    text_tokens, unit_tokens = zip(*batch)
    text_lens = [len(t) for t in text_tokens]
    unit_lens = [len(u) for u in unit_tokens]

    text_tokens_padded = nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=TEXT_PADDING_VALUE)
    unit_tokens_padded = nn.utils.rnn.pad_sequence(unit_tokens, batch_first=True, padding_value=UNIT_PADDING_VALUE)

    return text_tokens_padded, unit_tokens_padded, text_lens, unit_lens

class T2UPairConverter:
    def __init__(self, kmeans_uri=DEFAULT_KMEANS_URI, tokenizer_uri=DEFAULT_TOKENIZER_URI):
        # Unit extractor
        self.extractor = AlignmentExtractor(
            aligner_model_name_or_card="nar_t2u_aligner",
            unit_extractor_model_name_or_card="xlsr2_1b_v2",
            unit_extractor_output_layer=35,
            unit_extractor_kmeans_model_uri=kmeans_uri,
        )
        # Text tokenizer
        self.tokenizer = Tokenizer(tokenizer_uri)
        # Translator for T2S
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        self.translator = Translator(
            model_name_or_card="seamlessM4T_v2_large",
            vocoder_name_or_card="vocoder_v2",
            device=device,
            dtype=dtype,
            apply_mintox=True,
        )

    def get_unit_tokens(self, audio: str) -> list:
        # audio: str of audio path
        extractor = self.extractor
        audio_tensor = extractor.prepare_audio(audio)
        units = extractor.extract_units(audio_tensor)

        return units.tolist()
    
    def get_text_tokens(self, text) -> list:
        # text: str of text to be tokenized
        return self.tokenizer.encode(text, bos=False, eos=False)
    
    def convert_pair(self, text, audio) -> tuple[list, list]:
        # input: text script and audio path
        text_tokens = self.get_text_tokens(text)
        unit_tokens = self.get_unit_tokens(audio)
        return (text_tokens, unit_tokens)
    
    def convert_text(self, text: str) -> tuple[list, list]:
        # input: text script only
        # T2S
        source_language_code = target_language_code = "eng" # reference: https://github.com/facebookresearch/seamless_communication/blob/main/demo/expressive/utils.py#L1
        out_texts, out_audios = self.translator.predict(
            input=text,
            task_str="T2ST",
            src_lang=source_language_code,
            tgt_lang=target_language_code,
        )
        # TODO: add try catch 
        
        # out_text = str(out_texts[0])
        # out_wav = out_audios.audio_wavs[0]
        out_unit = out_audios.units[0]
        text_tokens = self.get_text_tokens(text)
        return (text_tokens, out_unit)
    
    def convert_text_prefix(self, text: str) -> tuple[list, list]:
        text_tokens = self.tokenizer.encode(text, bos=False, eos=False)
        text_lens = len(text_tokens)
        text_prefixes = []
        for i in range(1, text_lens+1):
            text_prefix = self.tokenizer.decode(text_tokens[:i])
            text_prefixes.append(text_prefix)
        
        return self.convert_text_batch(text_prefixes)
    
    def convert_text_batch(self, texts: list[str]) -> tuple[list[list], list[list]]:
        # input: text script only
        # T2S
        source_language_code = target_language_code = "eng" # reference: https://github.com/facebookresearch/seamless_communication/blob/main/demo/expressive/utils.py#L1
        try:
            out_texts, out_audios = self.translator.predict(
                input=texts,
                task_str="T2ST",
                src_lang=source_language_code,
                tgt_lang=target_language_code,
            )
        except RuntimeError as e:
            print(e)
            print("RE once. Probably CUDA OOM. Setting list to -1 and continue.")
            return ([-1 for _ in texts], [-1 for _ in texts])
        
        
        # out_text = str(out_texts[0])
        # out_wav = out_audios.audio_wavs[0]
        out_unit = out_audios.units
        text_tokens = [self.get_text_tokens(text) for text in texts]
        return (text_tokens, out_unit)
    
def convert_peoplesspeech_script(converter: T2UPairConverter, script_path, output_path,
                                  start_pos=None, end_pos=None, batch_size = 128):
    text = []
    unit = []
    
    with open(script_path, 'r') as f:
        data = f.readlines()
    n = len(data)

    assert not (start_pos is None) ^ (end_pos is None), "Either give both, or give none of start_pos and end_pos"
    if start_pos is None:
        start_pos = 0
        end_pos = n
    assert start_pos < end_pos, f"need start_pos < end_pos, current start_pos={start_pos}, end_pos={end_pos}"
    
    data = data[start_pos:end_pos]
    n = end_pos - start_pos

    for st in tqdm(range(0, n, batch_size)):
        ed = min(st + batch_size, n)
        text_tokens, unit_tokens = converter.convert_text_batch(data[st:ed])
        text.extend(text_tokens)
        unit.extend(unit_tokens)
        
    assert len(text) == len(unit) and len(unit) == n
    
    with open(output_path, 'wb') as f:
        pickle.dump({'n': n, 'text': text, 'unit':unit}, f)

def chunk(x: int):
    # CHUNK_CNT = 100
    # N = 3957126
    CHUNK_CNT = 25
    N = 1117560
    CHUNK_SIZE = ((N-1) // CHUNK_CNT) + 1 # 39572

    assert 0 <= x and x < CHUNK_CNT
    start = CHUNK_SIZE * x
    end = min(CHUNK_SIZE * (x + 1), N)
    
    return (start, end)

def load_data(data_folder, data_list=[0]):
    text_tokens = []
    unit_tokens = []
    for chunk in data_list:
        data_path = os.path.join(data_folder, f"chunk_{chunk}.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        text_tokens.extend(data['text'])
        unit_tokens.extend(data['unit'])
    
    text_tokens = [t for t, u in zip(text_tokens, unit_tokens) if len(u) <= 1024]
    unit_tokens = [u for u in unit_tokens if len(u) <= 1024]
    
    text_tokens = [torch.Tensor(t).long() for t in text_tokens]
    unit_tokens = [torch.Tensor(u).long() for u in unit_tokens]
    return text_tokens, unit_tokens

def inference(aligner, vocoder_agent, x = 0, with_prior=False):
    BATCH_SIZE = 1
    assert BATCH_SIZE == 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    data_folder = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/"
    train_text_tokens, train_unit_tokens = load_data(data_folder, [x])

    # train_text_tokens = train_text_tokens[:100]
    # train_unit_tokens = train_unit_tokens[:100]


    dataset = TextUnitDataset(train_text_tokens, train_unit_tokens)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    result = []
    result_idx = []
    # print(len(train_loader))
    # t = time.time()
    for batch_idx, (text_tokens, unit_tokens, text_lens, unit_lens) in tqdm(enumerate(train_loader)):
        try:
            # print("t0", time.time() - t)
            # t = time.time()
            text_tokens, unit_tokens = text_tokens.to(device), unit_tokens.to(device)
            
            text_tokens, unit_tokens = text_tokens[0], unit_tokens[0] # assert batch size == 1
            text_lens, unit_lens = text_lens[0], unit_lens[0]
            
            # print("t1", time.time() - t)
            # t = time.time()
            
            # generate audio wave from unit tokens
            states = AgentStates()
            states.source = unit_tokens.unsqueeze(0).unsqueeze(0)
            y = vocoder_agent.policy(states)
            waveform = torch.tensor(y.content.content, dtype=torch.float32, device='cpu').unsqueeze(0)
            
            # print("t2", time.time() - t)
            # t = time.time()
            
            # Align
            transcript = aligner.tokenizer.decode(text_tokens)
            
            # print("t3", time.time() - t)
            # t = time.time()
            
            segments = aligner.align(waveform, transcript)

            # print("t4", time.time() - t)
            # t = time.time()

            lengths = [s.length for s in segments]
            # print(len(lengths), text_lens)
            assert len(lengths) == text_lens

            lengths = scale_to_sum(lengths, unit_lens)
            result.append(lengths)
            result_idx.append(batch_idx)


            # print("t5", time.time() - t)
            # t = time.time()
        except Exception:
            print(batch_idx, (len(lengths), text_lens))
            pass
    
    prefix = "/scr-ssd" if os.path.exists("/scr-ssd") else "/scr/biggest"
    output_path = os.path.join(prefix, f"duyy/Alignment/result-ctc/chunk_{x}.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    
    idx_output_path = os.path.join(prefix, f"duyy/Alignment/result-ctc/idx_chunk_{x}.pkl")
    with open(idx_output_path, 'wb') as f:
        pickle.dump(result_idx, f)

def job(gpu_id: int, x: int):
    print(f"chunk {x} @ {gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job started at {now()}\n")

    torch.set_float32_matmul_precision('high')

    # Load models
    DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_URI, use_fast=False)
    aligner = ForcedAligner(tokenizer)
    vocoder_args = Namespace(
        vocoder_name='vocoder_v2',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        dtype=torch.float32,
        sample_rate=16000,
        tgt_lang='eng',
        vocoder_speaker_id=-1,
    )
    vocoder_agent = VocoderAgent(vocoder_args)

    inference(aligner, vocoder_agent, x, with_prior=False)

    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job finished at {now()}\n")

def get_hostname() -> str:
    result = subprocess.run(['hostname'], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()
def get_log_file(x):
    return f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment-CTC/result/logs/chunk_{x}.log"
def get_chunk(ngpu, start_chunk, end_chunk) -> list:
    assert start_chunk < end_chunk
    assert ngpu > 0
    ret = []
    for i in range(start_chunk, end_chunk):
        if len(ret) == ngpu:
            break
        if not os.path.isfile(get_log_file(i)):
            # Allocate job to process chunk i
            with open(get_log_file(i), 'w') as f:
                f.write(f"Chunk {i}: job allocated at {now()} on {get_hostname()}\n")
            ret.append(i)
    return ret
def distribute():
    parser = ArgumentParser()
    parser.add_argument("--ngpu", type=int, required=True)
    parser.add_argument("--start-chunk", type=int, default=0) # inclusive
    parser.add_argument("--end-chunk", type=int, default=25) # exclusive

    args = parser.parse_args()
    # List of parameters, one for each GPU
    params = get_chunk(args.ngpu, args.start_chunk, args.end_chunk)
    n = len(params)

    multiprocessing.set_start_method('spawn')

    while n > 0:
        # Create a process for each GPU
        processes = []
        for i in range(n): 
            p = multiprocessing.Process(target=job, args=(i, params[i]))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        params = get_chunk(args.ngpu, args.start_chunk, args.end_chunk)
        n = len(params)

if __name__ == "__main__":
    # converter = T2UPairConverter()

    # s = "Ultimately Titchener's ideas would form the basis of the short-lived psychological theory of structuralism."
    # text_tokens, out_unit = converter.convert_text_prefix(s)
    

    # import ipdb; ipdb.set_trace()

    # convert_peoplesspeech_script(converter,
    #                              script_path="/afs/cs.stanford.edu/u/duyy/data/downloads/peoplesspeech_script.txt",
    #                              output_path="/afs/cs.stanford.edu/u/duyy/data/downloads/peoplesspeech_TUpair.pkl")
    
    # import ipdb; ipdb.set_trace();

    assert os.path.exists("/scr-ssd") or os.path.exists("/scr/biggest")

    distribute()