from datasets import load_dataset
import tiktoken
from torch.utils.data import DataLoader, Dataset
import torch

class Wikitext2Dataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader(txt, batch_size, max_length, stride, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = Wikitext2Dataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers
    )
    return dataloader

def get_wikitext2_dataloaders(batch_size, max_length, stride):
    print("Downloading/Loading WikiText-2 dataset from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-v1")

    print("Concatenating splits...")
    train_text = "".join(dataset["train"]["text"])
    val_text = "".join(dataset["validation"]["text"])

    print("Building DataLoaders (this will take a moment to tokenize)...")
    train_loader = create_dataloader(train_text, batch_size, max_length, stride, shuffle=True)
    val_loader = create_dataloader(val_text, batch_size, max_length, stride, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    context_length = 1024
    batch_size = 4

    train_dl, val_dl = get_wikitext2_dataloaders(
        batch_size=batch_size,
        max_length=context_length, 
        stride=context_length
    )

    print(f"Train batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")

    x, y = next(iter(train_dl))
    print(f"Sample input shape: {x.shape}")