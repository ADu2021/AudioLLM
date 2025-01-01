## Convert from a pair of (text, audio) to a pair of (text tokens, unit tokens)
## Use conda env seamless @ /nlp/scr/duyy/miniconda3/envs/seamless/bin/python

from seamless_communication.inference import Translator
from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
from fairseq2.typing import Device
import torch
from llama.tokenizer import Tokenizer

import numpy
import pickle
from tqdm import tqdm

import multiprocessing
import os
from argparse import ArgumentParser
import datetime
import subprocess


def now():
    return datetime.datetime.now()

DEFAULT_KMEANS_URI = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"

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

def job(gpu_id: int, x: int):
    from train_forced_aligner_single_gpu import inference
    print(f"chunk {x} @ {gpu_id}")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job started at {now()}\n")

    inference(x, with_prior=False)

    with open(get_log_file(x), 'a') as f:
        f.write(f"Chunk {x}: job finished at {now()}\n")

def get_hostname() -> str:
    result = subprocess.run(['hostname'], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()
def get_log_file(x):
    return f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/result/logs/chunk_{x}.log"
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

    distribute()