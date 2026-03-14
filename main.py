import re

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import torch.linalg as linalg

torch.manual_seed(42)


def get_dataset():
    return "A quick brown fox jumped over a little lazy dog " * 4

# Making a tokenizer
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
        txt, batch_size=4, max_length=256,
        stride=128, shuffle=True, drop_last=True,
        num_workers=0
    ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader, tokenizer

# Make embeddings
def create_embeddings_layer(vocab_size, embedding_dim):
    return torch.nn.Embedding(vocab_size, embedding_dim)

# Make positional embeddings
def create_pos_embeddings_layer(context_len, embedding_dim):
    return torch.nn.Embedding(context_len, embedding_dim)

# Attention mechanisms
## Basic attention mechanism
##
## The goal is to create a embeddings for each word.
## It should represent how much attention is to be
## given to each word w.r.t. the current word.
## The token would contain a sequence of shape: [batch_size, context_len, embedding_dim]
## The output should be of the shape:           [batch_size, context_len, embedding_dim]
def basic_attention_mechanism(input):
    # 1. Find cosine similarity of each word with each other word
    cos_sim = input @ input.transpose(1, 2)
    print(f'{cos_sim.shape = }')

    # 2. Normalize with softmax
    attn_scores = torch.softmax(cos_sim, dim=-1)
    print(f'{attn_scores.shape = }')
    print(f'attn_scores sum:\n{attn_scores.sum(dim=-1)}')

    # 3. Compute outputs (The weighted sum of inputs)
    output = attn_scores @ input
    print(f'{output.shape = }')

    return output


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)


    def forward(self, x):
        ## Note: All the dimensions written ignore the batch
        ## size, torch would take care of batching itself. 
        # c => context_length
        # n => batch_size
        # h => embedding dim
        # o => output dim
        # x                             # c, h
        # Calculate attention score
        keys = self.W_key(x)           # c, o
        queries = self.W_query(x)      # c, o
        values = self.W_value(x)       # c, o

        # Attention score = query * key
        attn_scores = queries @ keys.transpose(1, 2)
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        # attn_weights                 # c, c
        # context_vec would be the weighted sum of values
        context_vec = attn_weights @ values
        return context_vec


class CausalAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.W_query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, output_dim, bias=qkv_bias)

        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            ).bool(),
        )

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        key = self.W_key(x)
        query = self.W_query(x)
        value = self.W_value(x)

        attn_score = query @ key.transpose(1, 2)
        attn_score.masked_fill_(
            self.mask[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(
            attn_score / query.shape[-1] ** 0.5,
            dim=-1,
        )

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ value
        return context_vec


def main():
    # Get dataset
    dataset = get_dataset()
    max_length = 4

    # Tokenizer the dataset
    dataloader, tokenizer = create_dataloader_v1(dataset, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)

    # Embedding layer
    vocab_size = tokenizer.n_vocab
    embedding_dim = 256
    embedding_layer = create_embeddings_layer(vocab_size=vocab_size, embedding_dim=embedding_dim)
    context_len = max_length
    pos_embedding_layer = create_pos_embeddings_layer(context_len=context_len, embedding_dim=embedding_dim)

    # Tokenize and get embeddings for the first batch
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    token_embeddings = embedding_layer(inputs)
    print(f'{token_embeddings.shape = }')

    
    pos_embeddings = pos_embedding_layer(torch.arange(token_embeddings.shape[-2]))
    print(f'{pos_embeddings.shape = }')

    input_embeddings = token_embeddings + pos_embeddings
    print(f'{input_embeddings.shape = }')

    # basic_attention_mechanism(input_embeddings)

    output_dim = 128
    self_attention = SelfAttention(embedding_dim, output_dim)
    sa_out = self_attention(input_embeddings)
    print(f'{sa_out.shape = }')
    sa_out_norm = linalg.norm(sa_out.flatten(), ord=2)
    print(f'{sa_out_norm = }')

    output_dim = 5
    self_attention = CausalAttention(embedding_dim, output_dim, context_len, 0.2)
    ca_out = self_attention(input_embeddings)
    print(f'{ca_out.shape = }')
    ca_out_norm = linalg.norm(ca_out.flatten(), ord=2)
    print(f'{ca_out_norm = }')
    


if __name__ == '__main__':
    main()
