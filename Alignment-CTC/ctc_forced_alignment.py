import torch
import torchaudio
from dataclasses import dataclass
from typing import List
import os
from transformers import AutoTokenizer

class ForcedAligner:
    @dataclass
    class Point:
        token_index: int
        time_index: int
        score: float

    @dataclass
    class Segment:
        label: str
        start: int
        end: int
        score: float

        def __repr__(self):
            return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

        @property
        def length(self):
            return self.end - self.start

    def __init__(self, tokenizer, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = tokenizer

        # Load model and labels
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().to(self.device)
        self.labels = bundle.get_labels()
        self.dictionary = {c: i for i, c in enumerate(self.labels)}
        self.blank_id = 0  # Usually 0 for CTC blank
        self.sample_rate = bundle.sample_rate
    
    def in_labels(self, x):
        return x in self.labels
    
    def align(self, waveform, transcript):
        # Process the waveform through the model to get emissions
        with torch.inference_mode():
            emissions, _ = self.model(waveform.to(self.device))
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()
        self.emission_length = emission.size(0)  # Store emission length for later use

        # Process the transcript into tokens
        tokens, labels_indices, token_boundaries = self._process_transcript(transcript)

        # Now compute trellis
        trellis = self.get_trellis(emission, labels_indices)

        # Backtrack to find the best path
        path = self.backtrack(trellis, emission, labels_indices)

        # Merge repeats to get segments
        segments = self.merge_tokens(path, token_boundaries, tokens)

        return segments

    def _process_transcript(self, transcript):
        # Tokenize the transcript using the LLaMA tokenizer
        token_ids = self.tokenizer.encode(transcript, add_special_tokens=False)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        # print("token_ids", token_ids)
        # print("tokens", tokens)
        # Build the label sequence and token boundaries
        label_sequence = []
        token_boundaries = []
        idx = 0  # position in label_sequence
        for token in tokens:
            # Decode the token to text
            token_text = self.tokenizer.convert_tokens_to_string([token])
            # print(token, token_text)
            # Remove leading/trailing spaces and newlines
            token_text = token_text.strip()

            # Skip empty tokens
            # if not token_text:
            #     continue

            # Map token text to uppercase to match model labels
            token_text_upper = token_text.upper()

            # Check if token_text_upper consists only of non-alphanumeric characters
            # if not any(c.isalnum() for c in token_text_upper):
            if not any(self.in_labels(c) for c in token_text_upper):
                # Token is non-alphanumeric, map to '|'
                token_labels = ['|']
            else:
                # Remove non-alphanumeric characters
                # token_text_upper = ''.join(filter(str.isalnum, token_text_upper))
                token_text_upper = ''.join(filter(self.in_labels, token_text_upper))
                # If the token is empty after cleaning, map to '|'
                if not token_text_upper:
                    token_labels = ['|']
                else:
                    # Split token_text_upper into characters
                    token_labels = list(token_text_upper)

            # Insert '|' as a token separator if needed
            if idx > 0:
                label_sequence.append('|')
                idx += 1

            # Append token labels
            label_sequence.extend(token_labels)
            token_start = idx
            idx += len(token_labels)
            token_end = idx
            token_boundaries.append((token_start, token_end))

        # Append final separator
        label_sequence.append('|')
        idx += 1

        # Map labels to indices
        try:
            labels_indices = [self.dictionary[label] for label in label_sequence]
        except KeyError as e:
            raise ValueError(f"Character '{e.args[0]}' not in model labels.")

        self.label_sequence = label_sequence  # Save for later use
        self.tokens = tokens
        self.token_boundaries = token_boundaries

        return tokens, labels_indices, token_boundaries

    def get_trellis(self, emission, tokens):
        num_frame = emission.size(0)
        num_labels = len(tokens)

        trellis = torch.full((num_frame, num_labels), -float('inf'))
        trellis[0, 0] = emission[0, tokens[0]]

        for t in range(1, num_frame):
            for j in range(num_labels):
                trellis[t, j] = trellis[t - 1, j] + emission[t, self.blank_id]
                if j > 0:
                    trellis[t, j] = max(
                        trellis[t, j],
                        trellis[t - 1, j - 1] + emission[t, tokens[j]]
                    )
        return trellis

    def backtrack(self, trellis, emission, tokens):
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [self.Point(j, t, emission[t, tokens[j]].exp().item())]
        while t > 0:
            if j > 0:
                p_stay = trellis[t - 1, j] + emission[t, self.blank_id]
                p_change = trellis[t - 1, j - 1] + emission[t, tokens[j]]
                if p_change > p_stay:
                    t -= 1
                    j -= 1
                else:
                    t -= 1
            else:
                # When j == 0, we can only stay at the first token
                t -= 1
            prob = emission[t, tokens[j]].exp().item()
            path.append(self.Point(j, t, prob))
        path.reverse()
        return path

    def merge_tokens(self, path, token_boundaries, tokens):
        segments = []
        token_idx = 0
        current_token = tokens[token_idx]
        start_time = None
        scores = []

        for point in path:
            if start_time is None:
                start_time = point.time_index

            # Check if we have moved to the next token
            if token_idx < len(token_boundaries) - 1 and point.token_index >= token_boundaries[token_idx + 1][0]:
                end_time = point.time_index
                avg_score = sum(scores) / len(scores) if scores else 0.0
                segments.append(
                    self.Segment(
                        label=current_token,
                        start=start_time,
                        end=end_time,
                        score=avg_score,
                    )
                )
                # Move to next token
                token_idx += 1
                if token_idx < len(tokens):
                    current_token = tokens[token_idx]
                else:
                    current_token = ''
                start_time = point.time_index
                scores = [point.score]
            else:
                scores.append(point.score)

        # Handle the last token
        if start_time is not None and token_idx < len(tokens):
            end_time = path[-1].time_index
            avg_score = sum(scores) / len(scores) if scores else 0.0
            segments.append(
                self.Segment(
                    label=current_token,
                    start=start_time,
                    end=end_time,
                    score=avg_score,
                )
            )

        return segments

def save_token_segments(audio_path, transcript, output_dir):
    # Initialize the LLaMA tokenizer
    DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_URI, use_fast=False)

    # Initialize the aligner
    aligner = ForcedAligner(tokenizer)

    # Load waveform
    waveform, sample_rate = torchaudio.load(audio_path)

    # Align
    segments = aligner.align(waveform, transcript)
    import ipdb; ipdb.set_trace();

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the ratio to map time indices to waveform sample indices
    emission_length = aligner.emission_length
    waveform_length = waveform.size(1)
    ratio = waveform_length / emission_length

    # Save each token segment as a separate audio file
    for idx, seg in enumerate(segments):
        start_sample = int(seg.start * ratio)
        end_sample = int(seg.end * ratio)
        token_waveform = waveform[:, start_sample:end_sample]
        token_label = seg.label.replace('|', '')  # Remove separator tokens
        # Clean up the token label for filename
        token_label_clean = ''.join(filter(str.isalnum, token_label))
        if not token_label_clean:
            token_label_clean = 'non_alnum'
        token_audio_path = os.path.join(output_dir, f'token_{idx}_{token_label_clean}.wav')
        torchaudio.save(token_audio_path, token_waveform, sample_rate)
        print(f"Saved token '{token_label}' to {token_audio_path}")

# DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/"
# tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_URI, use_fast=False)

# Usage example
if __name__ == "__main__":
    # Path to your .wav audio file
    audio_path = "/sailhome/duyy/data/AudioLLM/TTU/ckpt-2024-08-15_11-18-01-LLaMA-UnitEmbed-CANINE_sum-Continue-NGPU-2_BS-32_LR-1e-05-Warmup-3000-EPOCH-5-EMBED-1280-FFN-2048-DR-0.1-NH-8-NL-6-independent_tgt_mask-independent_mem_mask/bak/10.wav"

    # Transcript corresponding to the audio
    transcript = "Signature items were a manhole cover, , a slice, and a sunflower respectively."

    # Output directory to save token audio files
    output_dir = "/sailhome/duyy/temp"

    # Call the function to save token segments
    save_token_segments(audio_path, transcript, output_dir)
