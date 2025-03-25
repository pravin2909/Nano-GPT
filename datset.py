import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class ShakespeareDataset(Dataset):
    def __init__(self, seq_len=128, split="train", overlap=64):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        dataset = load_dataset("tiny_shakespeare", split=split)
        text = " ".join(dataset["text"])
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        print(f"Total tokens in dataset: {len(tokens)}")

        seq_len = min(seq_len, 1024)
        self.data = []
        for i in range(0, len(tokens) - seq_len, seq_len - overlap):
            chunk = tokens[i : i + seq_len]
            if len(chunk) == seq_len:
                self.data.append(chunk)

        print(f"Total sequences created: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1]).clone().detach()
        y = torch.tensor(self.data[idx][1:]).clone().detach()
        return x, y

def get_dataloader(batch_size=32, seq_len=128, split="train"):
    dataset = ShakespeareDataset(seq_len, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)