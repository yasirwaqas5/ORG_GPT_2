import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class GPTDataset(Dataset):
    def __init__(self, token_ids: List[int], block_size: int):
        if isinstance(token_ids, torch.Tensor):
            self.token_ids = token_ids.clone().detach().long()
        else:
            self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y


def create_datasets(token_ids: List[int], block_size: int, split_ratio: float = 0.9) -> Tuple[GPTDataset, GPTDataset]:
    n = int(split_ratio * len(token_ids))
    train_dataset = GPTDataset(token_ids[:n], block_size)
    val_dataset = GPTDataset(token_ids[n:], block_size)
    return train_dataset, val_dataset


def get_dataloader(dataset: GPTDataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
