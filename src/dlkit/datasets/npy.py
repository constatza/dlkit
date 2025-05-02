import torch
from numpy import load
from torch.utils.data import Dataset
from pathlib import Path
from pydantic import validate_call, FilePath


class NumpyDataset(Dataset):
    """
    Flexible dataset that loads from numpy file paths or wraps existing tensors.

    Args:
        features: Path to numpy file or a torch.Tensor of features.
        targets: Path to numpy file or a torch.Tensor of targets.
    """

    def __init__(
        self,
        features: Path,
        targets: Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        features_np = self.load(features)
        self.features = torch.from_numpy(features_np).type(dtype)
        if targets is not None:
            targets_np = self.load(targets)
            self.targets = torch.from_numpy(targets_np).type(dtype)
        else:
            self.targets = self.features

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @validate_call()
    def load(self, path: FilePath):
        return load(path)
