import torch
import torch.nn as nn
from typing import Optional
import json
from transformers import AutoConfig

# CANINE-C Embedding
# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/canine/modeling_canine.py#L191

# Support up to 16 hash functions.
_PRIMES = [31, 43, 59, 61, 73, 97, 103, 113, 137, 149, 157, 173, 181, 193, 211, 223]

class CanineEmbeddings(nn.Module):
    """Construct the character, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()

        self.config = config

        # character embeddings
        shard_embedding_size = config.hidden_size // config.num_hash_functions
        for i in range(config.num_hash_functions):
            name = f"HashBucketCodepointEmbedder_{i}"
            setattr(self, name, nn.Embedding(config.num_hash_buckets, shard_embedding_size))
        self.char_position_embeddings = nn.Embedding(config.num_hash_buckets, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def _hash_bucket_tensors(self, input_ids, num_hashes: int, num_buckets: int):
        """
        Converts ids to hash bucket ids via multiple hashing.

        Args:
            input_ids: The codepoints or other IDs to be hashed.
            num_hashes: The number of hash functions to use.
            num_buckets: The number of hash buckets (i.e. embeddings in each table).

        Returns:
            A list of tensors, each of which is the hash bucket IDs from one hash function.
        """
        if num_hashes > len(_PRIMES):
            raise ValueError(f"`num_hashes` must be <= {len(_PRIMES)}")

        primes = _PRIMES[:num_hashes]

        result_tensors = []
        for prime in primes:
            hashed = ((input_ids + 1) * prime) % num_buckets
            result_tensors.append(hashed)
        return result_tensors

    def _embed_hash_buckets(self, input_ids, embedding_size: int, num_hashes: int, num_buckets: int):
        """Converts IDs (e.g. codepoints) into embeddings via multiple hashing."""
        if embedding_size % num_hashes != 0:
            raise ValueError(f"Expected `embedding_size` ({embedding_size}) % `num_hashes` ({num_hashes}) == 0")

        hash_bucket_tensors = self._hash_bucket_tensors(input_ids, num_hashes=num_hashes, num_buckets=num_buckets)
        embedding_shards = []
        for i, hash_bucket_ids in enumerate(hash_bucket_tensors):
            name = f"HashBucketCodepointEmbedder_{i}"
            shard_embeddings = getattr(self, name)(hash_bucket_ids)
            embedding_shards.append(shard_embeddings)

        return torch.cat(embedding_shards, dim=-1)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self._embed_hash_buckets(
                input_ids, self.config.hidden_size, self.config.num_hash_functions, self.config.num_hash_buckets
            )

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.char_position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def extract_ckpt(in_path="/afs/cs.stanford.edu/u/duyy/data/downloads/canine-c/pytorch_model.bin", 
                 out_path="/afs/cs.stanford.edu/u/duyy/data/downloads/canine-c/canine_embedding.bin"):
    state_dict = torch.load(in_path)
    new_state_dict = {}
    # print(state_dict.keys())
    for k in state_dict.keys():
        if k.startswith("char_embeddings.") and not ("position_ids" in k):
            new_k = k.replace("char_embeddings.", "")
            new_state_dict[new_k] = state_dict[k]
    torch.save(new_state_dict, out_path)

    
if __name__ == "__main__":
    # extract_ckpt()
    # exit(0)
    config = AutoConfig.from_pretrained("google/canine-c")
    embed = CanineEmbeddings(config)
    ckpt_path = "/afs/cs.stanford.edu/u/duyy/data/downloads/canine-c/canine_embedding.bin"
    state_dict = torch.load(ckpt_path)
    embed.load_state_dict(state_dict)
    # import ipdb; ipdb.set_trace();
    text = "12"
    input_ids = [[ord(char) for char in text], [56, -1]]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_ids = input_ids.unsqueeze(0)
    out = embed(input_ids)
    print(out, out.shape)

    embed = 1