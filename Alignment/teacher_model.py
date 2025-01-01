import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math

from tqdm import tqdm
import pickle

class TransformerModel(nn.Module):
    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.pos_encoder = PositionalEncoding(dim_model, dropout)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.output_layer = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src = self.embedding(src) * math.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask, tgt_mask, None,
                                  src_padding_mask, tgt_padding_mask, None)
        output = self.output_layer(output)
        return output
    
    def encode(self, src, src_mask, src_key_padding_mask):
        return self.encoder(self.pos_encoder(self.embedding(src)), mask=src_mask, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        return self.decoder(self.pos_encoder(self.embedding(tgt)), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
def create_padding_mask(seq):
    # seq: [batch_size, seq_length]
    # Returns a mask where the padding values are marked as True.
    return (seq == -1)  # (batch_size, 1, 1, seq_length)
    # TODO: padding token is now -1, which is not useable

def generate(model, src, max_len=50, sos_token=0, eos_token=1):
    device = src.device
    src_mask = None  # Typically, no mask is needed for the encoder unless implementing specific behavior
    src_key_padding_mask = create_padding_mask(src) if src is not None else None

    # Encode the source sentence
    if src is not None:
        memory = model.transformer.encoder(model.pos_encoder(model.embedding(src)), mask=src_mask, src_key_padding_mask=src_key_padding_mask)
    else:
        memory = None

    # Start with the <sos> token
    ys = torch.tensor([[sos_token]], dtype=torch.long, device=device)

    for i in range(max_len - 1):  # Subtract 1 because we start with the <sos> token
        tgt_mask = create_look_ahead_mask(ys.size(1)).to(device)
        tgt_key_padding_mask = create_padding_mask(ys)  # This will all be False as no padding in ys

        out = model.transformer.decoder(model.pos_encoder(model.embedding(ys)), memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = model.output_layer(out[:, -1])
        next_word = out.argmax(1).item()

        ys = torch.cat([ys, torch.tensor([[next_word]], dtype=torch.long, device=device)], dim=1)
        
        # Stop if <eos> token is generated
        if next_word == eos_token:
            break

    return ys.squeeze()

## Start of Dataset
class Seq2SeqDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

# Example data
# NUM_TRAINING_DATA = int(2.5e4)
# SRC_SEQ_LEN = 20
# src_data = [torch.randint(2, 101, (SRC_SEQ_LEN,)) for _ in range(NUM_TRAINING_DATA)]
# for d in src_data:
#     d[0] = 0
#     d[-1] = 1
# # tgt_data = [torch.randint(0, 100, (100,)) for _ in range(100)]
# def dup(x):
#     ret = []
#     for i in range(1, SRC_SEQ_LEN-1):
#         target_int = torch.Tensor([(int(x[i]*17 + 4) % 100)+2]).long()
#         target_int = target_int.repeat(25)
#         ret.append(target_int)
    
#     ret = torch.cat(ret, dim=0)
#     noise = torch.randint(-5, 6, ret.shape)  # Adjust the range (-5, 6) as needed
#     ret = ret + noise
#     ret = (ret % 100) + 2

#     ret[0] = 0
#     ret[-1] = 1
        
#     return ret

# tgt_data = [dup(src) for src in src_data]

# real data
data_path = "/afs/cs.stanford.edu/u/duyy/data/downloads/peoplesspeech_TUpair/chunk_0.pkl"
with open(data_path, 'rb') as f:
    data = pickle.load(f)
n = data['n']
src_data = data['text']
tgt_data = data['unit']


train_dataset = Seq2SeqDataset(src_data, tgt_data)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

## End of Dataset

## Start of Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_TOKENS = 102  # Vocab size including special tokens
DIM_MODEL = 512
NUM_HEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DROPOUT = 0.1

# Model, loss, and optimizer
model = TransformerModel(NUM_TOKENS, DIM_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

def create_look_ahead_mask(size):
    # size: size of the sequence
    # Returns a mask to hide future tokens.
    mask = torch.triu(torch.ones((size, size)), diagonal=1).bool()
    return mask  # (size, size)

# Training
model.train()
for epoch in range(10):
    for src, tgt in tqdm(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[..., :-1]
        tgt_output = tgt[..., 1:]

        # Create padding masks
        src_padding_mask = create_padding_mask(src)
        tgt_padding_mask = create_padding_mask(tgt[:, :-1])  # Exclude last token for target padding mask
        
        # import ipdb; ipdb.set_trace();  
        
        src_mask = None  # These should be implemented based on your specific needs
        # tgt_mask = create_look_ahead_mask(tgt_input.size(1)).to(device)
        tgt_mask = model.transformer.generate_square_subsequent_mask(sz=tgt_input.size(1), device=tgt.device)

        predictions = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
        loss = loss_fn(predictions.reshape(-1, NUM_TOKENS), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

print("Training done.")

src = torch.tensor([[10, 20, 30, 40]], device=device)  # Example input sequence
generated_seq = generate(model, src, max_len=200, sos_token=0, eos_token=1)  # EOS token assumed to be 2
print("Generated sequence:", generated_seq)

import ipdb; ipdb.set_trace();
## End of Training

# def translate(model, src_sentence, max_len=10):
#     model.eval()
#     with torch.no_grad():
#         src = torch.tensor(src_sentence).unsqueeze(0).to(device)
#         src_mask = None  # Implement as needed

#         memory = model.transformer.encoder(model.pos_encoder(model.embedding(src)), mask=src_mask)
#         ys = torch.ones(1, 1).fill_(1).type(torch.long).to(device)  # assuming 1 is the start token

#         for i in range(max_len-1):
#             tgt_mask = (generate_square_subsequent_mask(ys.size(1))
#                         .type(torch.bool)).to(device)
#             out = model.transformer.decoder(model.pos_encoder(model.embedding(ys)), memory, tgt_mask)
#             out = model.output_layer(out)
#             next_word = out[:, -1].argmax(1).item()
#             ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(device)], dim=1)

#             if next_word == 2:
#                 break  # assuming 2 is the end token

#         return ys.cpu().numpy()

# # Usage
# translated_sentence = translate(model, [12, 24, 36, 48, 60])  # example source sentence token IDs
# print("Translated sentence:", translated_sentence)
