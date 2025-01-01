from turtle import forward
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Union
import numpy as np
import numpy.typing as npt
import torch.nn.functional as F
from torch import Tensor

class ConvSequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size):
        super(ConvSequenceModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # Ensuring that the convolution output has the same length as the input sequence
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_features=num_filters),
            nn.ReLU()
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_features=num_filters),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(num_features=num_filters),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch_size, sequence_length)

        # Embedding layer
        x = self.embedding(x)  # Output shape: (batch_size, sequence_length, embedding_dim)

        # Transpose to fit Conv1d input requirements (batch_size, channels, length)
        x = x.transpose(1, 2)

        # Applying convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Transpose back to (batch_size, sequence_length, num_filters)
        x = x.transpose(1, 2)

        return x

class TransformerSequenceModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, nhead, num_encoder_layers, dim_feedforward, padding_idx=None):
        super(TransformerSequenceModel, self).__init__()
        
        self.padding_idx = padding_idx if padding_idx is not None else (vocab_size-1)
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=self.padding_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

    def forward(self, x, padding_mask):
        # x: (batch_size, sequence_length)
        # padding_mask should be (x == self.padding_idx)

        # Embedding layer
        x = self.embedding(x)  # Output shape: (batch_size, sequence_length, embedding_dim)
        
        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # Output shape: (batch_size, sequence_length, embedding_dim)

        return x
    
    def get_padding_mask(self, x):
        return x == self.padding_idx

@dataclass
class EncoderConfig(ABC):
    vocab_size: int
@dataclass
class AttnEncoderConfig(EncoderConfig):
    vocab_size: int = 4096
    embedding_dim: int = 1024
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 8192

def _monotonic_alignment_search(
    attn_lprob: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # https://arxiv.org/abs/2005.11129
    T_feat = attn_lprob.shape[0]
    T_text = attn_lprob.shape[1]
    Q = np.full((T_text, T_feat), fill_value=-np.inf)

    log_prob = attn_lprob.transpose(1, 0)  # -> (T_text, T_feat)
    # 1.  Q <- init first row for all j
    for j in range(T_feat):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_feat):
        for i in range(1, min(j + 1, T_text)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_feat,), fill_value=T_text - 1)
    for j in range(T_feat - 2, -1, -1):  # T_feat-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    # import ipdb; ipdb.set_trace()
    return A
def viterbi_decode(
    attn_lprob: Tensor, text_lengths: Tensor, feat_lengths: Tensor
) -> Tensor:
    """Extract duration from an attention probability matrix
    Reference: https://github.com/facebookresearch/seamless_communication/blob/main/src/seamless_communication/models/aligner/model.py#L246

    Args:
        attn_lprob (Tensor): Batched log probability of attention
            matrix (B, T_feat, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feat_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `attn_lprob` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    attn_lprob = attn_lprob.squeeze()

    B = attn_lprob.size(0)
    T_text = attn_lprob.size(2)
    device = attn_lprob.device

    durations = torch.zeros((B, T_text), device=device, dtype=torch.long)
    for b in range(B):
        assert feat_lengths[b] > 0
        assert text_lengths[b] > 0
        cur_log_p_attn = attn_lprob[b, : feat_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search(
            cur_log_p_attn.float().detach().cpu().numpy()
        )
        _durations = np.bincount(viterbi)
        durations[b, : len(_durations)] = torch.from_numpy(_durations).to(device)

    return durations
    
from scipy.special import binom, betaln
import numpy as np

def beta_binomial_pmf(n, k, alpha, beta):
    """
    Compute the probability mass function of the beta-binomial distribution.
    
    :param n: number of trials
    :param k: number of successes
    :param alpha: alpha parameter of the Beta distribution
    :param beta: beta parameter of the Beta distribution
    :return: probability of observing k successes out of n trials
    """
    if alpha == 0:
        return 1 if (k == 0) else 0
    log_pmf = (np.log(binom(n, k)) +
               betaln(k + alpha, n - k + beta) -
               betaln(alpha, beta))
    return np.exp(log_pmf)
def get_prior(text_length, unit_length, w=1.0):
    prior = [[beta_binomial_pmf(text_length, k, w*t, w*(unit_length-t+1)) for k in range(text_length)] for t in range(unit_length)]
    # shape [unit_length, text_length]
    return prior
class Aligner(nn.Module):
    def __init__(self, text_vocab_size=128_257, unit_encoder_config=10_001):
        super(Aligner, self).__init__()
        
        self.text_encoder = Aligner.get_encoder_from_config(text_vocab_size)
        self.unit_encoder = Aligner.get_encoder_from_config(unit_encoder_config)

    @classmethod
    def get_encoder_from_config(cls, vocab_size):
        embedding_dim = 1024 # Size of each embedding vector
        nhead = 8          # Number of attention heads
        num_encoder_layers = 3  # Number of transformer encoder layers
        dim_feedforward = 2048   # Dimension of the feedforward network in transformer
        return TransformerSequenceModel(vocab_size, embedding_dim, nhead, num_encoder_layers, dim_feedforward)
        
    debug = False
        
    def forward(self, text_tokens, unit_tokens, with_prior=False, text_token_lengths=None, unit_token_lengths=None):
        """
        text_tokens: shape [batch_size, max_text_sequence_length] with padding text_vocab_size-1
        unit_tokens: shape [batch_size, max_unit_sequence_length] with padding unit_vocab_size-1
        
        {text,unit}_token_lengths: (Optional) shape [batch_size]
        """

        text_padding_mask = self.text_encoder.get_padding_mask(text_tokens) # [batch_size, max_text_sequence_length]
        unit_padding_mask = self.unit_encoder.get_padding_mask(unit_tokens) # [batch_size, max_unit_sequence_length]

        text_feature = self.text_encoder(text_tokens, text_padding_mask) # output shape: [batch_size, max_text_sequence_length, embedding_dim]
        unit_feature = self.unit_encoder(unit_tokens, unit_padding_mask) # output shape: [batch_size, max_unit_sequence_length, embedding_dim]

        temp = 0.0005 # Reference: https://github.com/NVIDIA/radtts/blob/07759cd474458f46db45cab975a85ba21b7fee0a/common.py#L902
        dist = unit_feature.unsqueeze(2) - text_feature.unsqueeze(1) # [batch_size, max_unit_sequence_length, max_text_sequence_length, embedding_dim]
        dist = -temp * (dist * dist).sum(-1)# [batch_size, max_unit_sequence_length, max_text_sequence_length]

        dist_padding_mask = unit_padding_mask.unsqueeze(2) | text_padding_mask.unsqueeze(1) # [batch_size, max_unit_sequence_length, max_text_sequence_length]
        filled_dist = dist.masked_fill(dist_padding_mask, -np.inf)
        # import ipdb; ipdb.set_trace()

        log_prob = F.log_softmax(filled_dist, dim=-1)
        # with_prior = False
        if with_prior:
            assert (text_token_lengths is not None) and (unit_token_lengths is not None)
            batch_size = log_prob.shape[0]
            device = log_prob.device
            prior_shaped = torch.zeros(log_prob.shape).to(device)

            for b in range(batch_size):
                prior = torch.Tensor(get_prior(text_token_lengths[b], unit_token_lengths[b])).to(device)
                prior_shaped[b, : prior.shape[0], : prior.shape[1]] = prior
            
            eps = 1e-8
            prior_softmax_prob = log_prob + torch.log(prior_shaped + eps)
            # if self.debug:
            # import ipdb; ipdb.set_trace();
            # prior_softmax_prob = softmax_prob
            log_prob = prior_softmax_prob
            # self.debug = True
        
        final_log_prob = log_prob.clone()
            
        return final_log_prob

if __name__ == "__main__":
    import ipdb; ipdb.set_trace()
    pass
