import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from forced_aligner import Aligner, viterbi_decode
from CTC import ForwardSumLoss  # Import your ForwardSumLoss here

from datetime import datetime

import pickle
import wandb
from tqdm import tqdm
WANDB = True

TEXT_PADDING_VALUE = 128_256
UNIT_PADDING_VALUE = 10_000

DATASET = "mls"
if DATASET == "commonvoice":
    DATA_FOLDER = "/afs/cs.stanford.edu/u/duyy/data/downloads/commonvoice_TUpair/"
elif DATASET == "mls":
    DATA_FOLDER = "/scr-ssd/duyy/MLS_Full_En_TUPair/"

class TextUnitDataset(Dataset):
    """ Dataset class to handle text and unit token pairs. """
    def __init__(self, text_tokens, unit_tokens):
        self.text_tokens = text_tokens
        self.unit_tokens = unit_tokens

    def __len__(self):
        return len(self.text_tokens)

    def __getitem__(self, idx):
        return torch.tensor(self.text_tokens[idx], dtype=torch.long),\
               torch.tensor(self.unit_tokens[idx], dtype=torch.long)

def collate_fn(batch):
    """ Collate function to pad the sequences in the batch. """
    text_tokens, unit_tokens = zip(*batch)
    text_lens = [len(t) for t in text_tokens]
    unit_lens = [len(u) for u in unit_tokens]

    text_tokens_padded = nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=TEXT_PADDING_VALUE)
    unit_tokens_padded = nn.utils.rnn.pad_sequence(unit_tokens, batch_first=True, padding_value=UNIT_PADDING_VALUE)

    return text_tokens_padded, unit_tokens_padded, text_lens, unit_lens

def train(model, device, train_loader, val_loader, optimizer, epoch, rank, run_name, with_prior, validate_every_n_batches=250):
    model.train()
    loss_function = ForwardSumLoss()
    total_batches = len(train_loader.dataset) // (train_loader.batch_size * dist.get_world_size())

    for batch_idx, (text_tokens, unit_tokens, text_lens, unit_lens) in enumerate(train_loader):
        text_tokens, unit_tokens = text_tokens.to(rank), unit_tokens.to(rank)

        optimizer.zero_grad()
        distances = model(text_tokens, unit_tokens, with_prior=with_prior, text_token_lengths=text_lens, unit_token_lengths=unit_lens)
        distances = distances.unsqueeze(1)
        # import ipdb; ipdb.set_trace();
        loss = loss_function(distances, torch.tensor(text_lens), torch.tensor(unit_lens))
        loss.backward()
        optimizer.step()

        if rank == 0 and WANDB:
            wandb.log({"train_loss": loss.item(), "epoch": epoch + batch_idx / total_batches})

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(text_tokens)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        if batch_idx % validate_every_n_batches == 0:
            model.eval()
            val_loss = 0
            num_batches = 0
            with torch.no_grad():
                for val_text_tokens, val_unit_tokens, val_text_lens, val_unit_lens in val_loader:
                    val_text_tokens, val_unit_tokens = val_text_tokens.to(device), val_unit_tokens.to(device)
                    val_distances = model(val_text_tokens, val_unit_tokens, with_prior=with_prior, text_token_lengths=val_text_lens, unit_token_lengths=val_unit_lens)
                    val_distances = val_distances.unsqueeze(1)
                    val_loss += loss_function(val_distances, torch.tensor(val_text_lens), torch.tensor(val_unit_lens)).item()
                    num_batches += 1
            val_loss /= num_batches
            print(f'Validation Loss after {batch_idx} batches: {val_loss:.6f}')
            
            if wandb.run:
                wandb.log({"val_loss": val_loss, "epoch": epoch + batch_idx / total_batches})

            model.train()  # Switch back to training mode

    # Save the model
    if rank == 0:
        print("Saving...")
        SAVE_PATH = f"/sailhome/duyy/data/checkpoints/ForcedAligner/ckpt-{run_name}/"
        SAVE_NAME = f"epoch-{epoch}.bin"
        os.makedirs(SAVE_PATH, exist_ok = True) 
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, SAVE_NAME))

MAX_UNIT_SEQ_LEN = 256
def load_data(data_folder, data_list=[0]):
    text_tokens = []
    unit_tokens = []
    for chunk in tqdm(data_list):
        if DATASET == "commonvoice":
            data_path = os.path.join(data_folder, f"chunk_{chunk}.pkl")
        elif DATASET == "mls":
            data_path = os.path.join(data_folder, f"train-{str(chunk).zfill(5)}-of-04096.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        data['text'] = [t for t, u in zip(data['text'], data['unit']) if len(u) <= 768]
        data['unit'] = [u for u in data['unit'] if len(u) <= 768]
        text_tokens.extend(data['text'])
        unit_tokens.extend(data['unit'])
    
    # text_tokens = [t for t, u in zip(text_tokens, unit_tokens) if len(u) <= 1024]
    # unit_tokens = [u for u in unit_tokens if len(u) <= 1024]
    
    # text_tokens = [torch.tensor(t, dtype=torch.long) for t in text_tokens]
    # unit_tokens = [torch.tensor(u, dtype=torch.long) for u in unit_tokens]
    # import ipdb; ipdb.set_trace();

    print(f"Data Loaded.")
    return text_tokens, unit_tokens

def parse_state_dict(state_dict):
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        if new_key.startswith('module.'):
            new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict
def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()
    print(f"Running DDP training on rank {rank}.")

    # Training Config
    BATCH_SIZE = 64
    EPOCH = 1
    LR = 5e-5
    WITH_PRIOR = True
    # CONTINUE_TRAINING = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-24_00-50-38-NGPU-2_BS-64_LR-5e-05-EPOCH-4-with-prior/epoch-3.bin"
    CONTINUE_TRAINING = None #"/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-25_00-02-53-NGPU-4_BS-64_LR-5e-05-EPOCH-5-continue/epoch-4.bin"

    GPU_COUNT = torch.cuda.device_count()    

    
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = DATASET
    run_name = f"{current_time}-{dataset_name}-NGPU-{GPU_COUNT}_BS-{BATCH_SIZE}_LR-{LR}-EPOCH-{EPOCH}{'-with-prior' if WITH_PRIOR else ''}{'-continue' if CONTINUE_TRAINING is not None else ''}"
    
    if rank == 0 and WANDB:
        # Initialize wandb
        wandb.init(
            project="text-unit-alignment", 
            name=run_name,
            config={
                "learning_rate": LR,
                "epochs": EPOCH,
                "batch_size": BATCH_SIZE,
                "dataset": DATASET,
                "continue": CONTINUE_TRAINING,
                #TODO: model config
            }
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    # Example data
    # text_tokens = [torch.tensor([1, 2, 3, 4]), torch.tensor([1, 2, 3])] * 100
    # unit_tokens = [torch.tensor([5, 6, 7, 8, 9]), torch.tensor([5, 6, 7])] * 100
    
    data_folder = DATA_FOLDER
    train_text_tokens, train_unit_tokens = load_data(data_folder, range(4095)) # 24 for commonvoice and 4095 for mls
    
    val_text_tokens, val_unit_tokens = load_data(data_folder, [4095])
    val_text_tokens = val_text_tokens[:1280]
    val_unit_tokens = val_unit_tokens[:1280]


    train_dataset = TextUnitDataset(train_text_tokens, train_unit_tokens)
    train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, collate_fn=collate_fn)

    val_dataset = TextUnitDataset(val_text_tokens, val_unit_tokens)
    val_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, collate_fn=collate_fn)

    model = Aligner()
    if CONTINUE_TRAINING:
        model_path = CONTINUE_TRAINING
        state_dict = torch.load(model_path)
        state_dict = parse_state_dict(state_dict)
        model.load_state_dict(state_dict)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    print("Converted model to DDP model")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    num_epochs = EPOCH
    print("Start training")
    for epoch in range(num_epochs):
        train(model, device, train_loader, val_loader, optimizer, epoch, rank, run_name, WITH_PRIOR)

    if rank == 0:
        if WANDB:
            wandb.finish()

def inference(x = 0, with_prior=False):
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    data_folder = DATA_FOLDER
    train_text_tokens, train_unit_tokens = load_data(data_folder, [x])


    dataset = TextUnitDataset(train_text_tokens, train_unit_tokens)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    model = Aligner()

    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-24_00-50-38-NGPU-2_BS-64_LR-5e-05-EPOCH-4-with-prior/epoch-3.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-25_00-02-53-NGPU-4_BS-64_LR-5e-05-EPOCH-5-continue/epoch-4.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-26_17-43-12-NGPU-2_BS-64_LR-5e-05-EPOCH-10-paired/epoch-9.bin"
    # model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-07-26_17-43-48-NGPU-2_BS-64_LR-5e-05-EPOCH-10/epoch-9.bin"
    
    model_path = "/afs/cs.stanford.edu/u/duyy/data/checkpoints/ForcedAligner/ckpt-2024-08-18_16-09-06-mls-NGPU-2_BS-64_LR-5e-05-EPOCH-5-continue/epoch-2.bin"
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

    output_path = f"/scr-ssd/duyy/Alignment/result/chunk_{x}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
        
if __name__ == "__main__":
    main()