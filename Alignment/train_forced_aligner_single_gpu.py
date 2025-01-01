import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from forced_aligner import Aligner, viterbi_decode
from CTC import ForwardSumLoss  # Import your ForwardSumLoss here

import pickle
import wandb
from tqdm import tqdm
WANDB = False

TEXT_PADDING_VALUE = 128_256
UNIT_PADDING_VALUE = 10_000

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

def train(model, device, train_loader, val_loader, optimizer, epoch):
    model.train()
    loss_function = ForwardSumLoss()
    for batch_idx, (text_tokens, unit_tokens, text_lens, unit_lens) in enumerate(train_loader):
        text_tokens, unit_tokens = text_tokens.to(device), unit_tokens.to(device)

        optimizer.zero_grad()
        distances = model(text_tokens, unit_tokens)
        distances = distances.unsqueeze(1)
        
        loss = loss_function(distances, torch.tensor(text_lens), torch.tensor(unit_lens))
        loss.backward()
        optimizer.step()

        if WANDB:
            wandb.log({"loss": loss.item()})  # Log training loss

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(text_tokens), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    print("Saving...")
    SAVE_PATH = "/sailhome/duyy/data/checkpoints/ForcedAligner/ckpt.bin"
    torch.save(model.state_dict(), SAVE_PATH)

MAX_UNIT_SEQ_LEN = 256
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

def demo(text_tokens, unit_tokens, align, text_len, unit_len, id):
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
    
    text_tokens = text_tokens[:text_len]
    unit_tokens = unit_tokens[:unit_len]
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
            output_folder = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/demo/demo-{id}"
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f"{i}_{text_tokens[i]}_{text_token}.wav")
            # import ipdb; ipdb.set_trace()
            torchaudio.save(output_path, content, sample_rate=args.sample_rate)
        except:
            print("no")
            pass
        start_pos = end_pos
    

def parse_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict
def inference(x = 0, with_prior=False):
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    data_folder = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/"
    train_text_tokens, train_unit_tokens = load_data(data_folder, [x])


    dataset = TextUnitDataset(train_text_tokens, train_unit_tokens)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = Aligner()

    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-24_00-50-38-NGPU-2_BS-64_LR-5e-05-EPOCH-4-with-prior/epoch-3.bin"
    model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-25_00-02-53-NGPU-4_BS-64_LR-5e-05-EPOCH-5-continue/epoch-4.bin"
    
    # ablation
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-26_17-43-12-NGPU-2_BS-64_LR-5e-05-EPOCH-10-paired/epoch-9.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-26_17-43-48-NGPU-2_BS-64_LR-5e-05-EPOCH-10/epoch-9.bin"
    
    # qwen
    model_path = "/sailhome/duyy/data/checkpoints/ForcedAligner-qwen/ckpt-2024-09-30_04-51-28-commonvoice-NGPU-2_BS-64_LR-5e-05-EPOCH-10-continue/epoch-9.bin"

    state_dict = torch.load(model_path)
    state_dict = parse_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model = model.to(device)

    model.eval()

    result = []

    for batch_idx, (text_tokens, unit_tokens, text_lens, unit_lens) in tqdm(enumerate(train_loader)):
        text_tokens, unit_tokens = text_tokens.to(device), unit_tokens.to(device)

        distances = model(text_tokens, unit_tokens, with_prior=with_prior, text_token_lengths=text_lens, unit_token_lengths=unit_lens)
        batch_result = viterbi_decode(distances, text_lens, unit_lens) # (B, T_text)
        # import ipdb; ipdb.set_trace();
        result.extend(batch_result.tolist())

        # for i in range(0, batch_result.shape[0]):
        #     demo(text_tokens[i], unit_tokens[i], batch_result[i], text_lens[i], unit_lens[i], i)
        # exit(0)

    output_path = f"/afs/cs.stanford.edu/u/duyy/data/AudioLLM/Alignment/result/chunk_{x}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)


def main():
    BATCH_SIZE = 64
    EPOCH = 1
    LR = 1e-5

    if WANDB:
        # Initialize wandb
        wandb.init(
            project="text-unit-alignment", 
            config={
                "learning_rate": LR,
                "epochs": EPOCH,
                "batch_size": BATCH_SIZE,
                #TODO: model config
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # Example data
    # text_tokens = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3])] * 100
    # unit_tokens = [torch.tensor([5, 6, 7, 8, 9]), torch.tensor([5, 6, 7])] * 100
    
    data_folder = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/"
    train_text_tokens, train_unit_tokens = load_data(data_folder, range(24))
    
    val_text_tokens, val_unit_tokens = load_data(data_folder, [24])
    val_text_tokens = val_text_tokens[:1000]
    val_unit_tokens = val_unit_tokens[:1000]


    dataset = TextUnitDataset(train_text_tokens, train_unit_tokens)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    val_dataset = TextUnitDataset(val_text_tokens, val_unit_tokens)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Aligner().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    num_epochs = EPOCH
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, val_loader, optimizer, epoch)

    if WANDB:
        wandb.finish()

if __name__ == "__main__":
    # main()
    inference(with_prior=False)
