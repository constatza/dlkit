import torch
from torch.utils.data import DataLoader


def dataloader_to_tensor(dataloader: DataLoader):
    tensors_features = []
    tensors_targets = []
    for batch in dataloader:
        x, y = batch
        tensors_features.append(x)
        tensors_targets.append(y)
    return torch.cat(tensors_features), torch.cat(tensors_targets)
