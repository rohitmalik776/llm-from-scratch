import re

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


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
    


if __name__ == '__main__':
    main()
