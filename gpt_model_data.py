import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


THE_VERDICT_DATASET_PATH = './the_verdict.txt'

def load_text_dataset(path):
    with open(path, 'r', encoding='utf-8') as file:
        txt_data = file.read()

    return txt_data

def tokenize_dataset(text, tokenizer: tiktoken.Encoding):
    token_ids = tokenizer.encode(text=text, allowed_special={'<|endoftext|>'})

    print(f'Total tokens in dataset: {len(token_ids)}')

    return token_ids

def train_val_split(token_ids, train_ratio):
    train_len = int(len(token_ids) * train_ratio)

    train_ids = token_ids[:train_len]
    val_ids = token_ids[train_len:]

    return train_ids, val_ids



class TextDataset(Dataset):

    def __init__(self, token_ids, max_length, stride):
        super().__init__()
        self.max_length = max_length
        self.stride = stride
        self.token_ids = token_ids


    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length

        return torch.tensor(self.token_ids[start : end]), torch.tensor(self.token_ids[start + 1 : end + 1])

    def __len__(self):
        return ((len(self.token_ids) - self.max_length - 1) // self.stride) + 1


def get_verdict_dataloaders(
        tokenizer: tiktoken.Encoding, 
        train_ratio: float, 
        batch_size: int, 
        max_length: int,
        stride: int,
        shuffle_train=True, 
        num_workers=0,
        pin_memory=True
    ):
    text = load_text_dataset(path=THE_VERDICT_DATASET_PATH)
    token_ids = tokenize_dataset(text, tokenizer=tokenizer)
    train_ids, val_ids = train_val_split(token_ids, train_ratio)

    train_ds = TextDataset(train_ids, max_length, stride)
    val_ds = TextDataset(val_ids, max_length, stride)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader
