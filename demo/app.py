# Please use torchrun to run this script

import gradio as gr
import math
import numpy as np
import time
import io
import wave
import torch

from model import load_model, load_vocoder_agent
from model import greedy_decode, unit_to_audio, generate_square_subsequent_mask
from model import SEP_TOKEN

torch.set_float32_matmul_precision('high')

SAMPLE_RATE = 16000
MAX_SEQ_LEN = 1020

STREAMING = True

DEMO_VALUE = "In the heart of a bustling city, nestled between gleaming skyscrapers, there lay a quaint little park known for its ancient oak tree. This tree, rumored to be over a century old, was where locals and tourists alike would find solace from the urban rush. One sunny afternoon, a young boy named Leo wandered into the park."

# model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/TTU/ckpt-2024-08-15_11-18-01-LLaMA-UnitEmbed-CANINE_sum-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-4.bin"
model_path = "/sailhome/duyy/data/checkpoints/TTU/ckpt-2024-09-26_08-48-39-llama3-UnitEmbed-CANINE_sum-Continue-Double-NGPU-1_BS-32_LR-1e-05-Warmup-3000-EPOCH-3-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/epoch-2.bin"
model = load_model(model_path)
tokenizer = model.text_encoder.model.tokenizer
vocoder_agent = load_vocoder_agent()

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=24000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()

def process_text(text_input):
    yield wave_header_chunk(sample_rate=SAMPLE_RATE)
    volume = 10000
    
    src = tokenizer.encode(text_input, eos=False, bos=False)
    src = src[:MAX_SEQ_LEN]
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
    src_mask = generate_square_subsequent_mask(src.size(1))
    
    start_time = time.time()
    for unit_chunk in greedy_decode(model, src, src_mask, max_len=1000, start_symbol=SEP_TOKEN, sep_token=SEP_TOKEN):
        audio_chunk = unit_to_audio(vocoder_agent, unit_chunk)
        audio_chunk = (audio_chunk * volume).squeeze()
        np_audio_chunk = audio_chunk.numpy().astype(np.int16)
        yield np_audio_chunk.tobytes()

def process_text_mp3(text_input):
    # yield wave_header_chunk(sample_rate=SAMPLE_RATE)
    volume = 10000
    
    src = tokenizer.encode(text_input, eos=False, bos=False)
    src = src[:MAX_SEQ_LEN]
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
    src_mask = generate_square_subsequent_mask(src.size(1))
    
    start_time = time.time()
    for unit_chunk in greedy_decode(model, src, src_mask, max_len=1000, start_symbol=SEP_TOKEN, sep_token=SEP_TOKEN):
        audio_chunk = unit_to_audio(vocoder_agent, unit_chunk)
        audio_chunk = (audio_chunk * volume).squeeze()
        np_audio_chunk = audio_chunk.numpy().astype(np.int16)

        # test for mp3
        from pydub import AudioSegment
        wav_audio = AudioSegment(np_audio_chunk.tobytes(), frame_rate=16_000, sample_width=2, channels=1)
        # Convert to MP3 bytes
        mp3_buffer = io.BytesIO()
        wav_audio.export(mp3_buffer, format="mp3")
        mp3_bytes = mp3_buffer.getvalue()
        mp3_buffer.close()
        print("mp3_bytes:", mp3_bytes)
        yield mp3_bytes
        
def process_text_non_stream(text_input):
    out = wave_header_chunk(sample_rate=SAMPLE_RATE)
    volume = 10000
    
    src = tokenizer.encode(text_input, eos=False, bos=False)
    src = src[:MAX_SEQ_LEN]
    src = torch.tensor(src, dtype=torch.long).unsqueeze(0)
    src_mask = generate_square_subsequent_mask(src.size(1))
    
    start_time = time.time()
    for unit_chunk in greedy_decode(model, src, src_mask, max_len=1000, start_symbol=SEP_TOKEN, sep_token=SEP_TOKEN):
        audio_chunk = unit_to_audio(vocoder_agent, unit_chunk)
        audio_chunk = (audio_chunk * volume).squeeze()
        np_audio_chunk = audio_chunk.numpy().astype(np.int16)
        out += np_audio_chunk.tobytes()
    return out

def yield_from_stream():
    yield wave_header_chunk()
    volume = 10000  # range [0.0, 1.0]
    fs = 48000  # sampling rate, Hz, must be integer
    duration = 0.25  # in seconds, may be float
    f = 440.0  # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    num_samples = int(fs * duration)

    while True:
        if f == 440.0:
            f = 880.0
        else:
            f = 440.0
        samples = np.array(
            [
                int(volume * math.sin(2 * math.pi * k * f / fs))
                for k in range(0, num_samples)
            ],
            dtype=np.int16,
        )
        print((48000, samples))
        print("before yield", time.time())
        yield samples.tobytes()
        print("after yield", time.time())
        time.sleep(0.5)


with gr.Blocks() as demo:
    text_input = gr.Textbox(label="Text Script", value=DEMO_VALUE)
    out = gr.Audio(
        streaming=True,
        autoplay=True,
        interactive=False,
    )
    start = gr.Button("Start Stream")

    start.click(process_text_mp3, [text_input], [out])

DEBUG = False
if __name__ == "__main__":
    if DEBUG == False:
        # launch demo
        demo.launch(share=True, debug=True)
    else:
        prev_time = time.time()
        for x in process_text(DEMO_VALUE):
            print(time.time() - prev_time)
            prev_time = time.time()