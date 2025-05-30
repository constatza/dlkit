from asyncio import Protocol
from pathlib import Path

import torch
from numpy import load
from dlkit.datatypes.dataset import Shape
from .base import BaseDataset


class ArrayDataset(Protocol):
    """Protocol for a dataset that loads from numpy file paths or wraps existing tensors."""

    features: torch.Tensor
    targets: torch.Tensor
    shape: Shape


class NumpyDataset(BaseDataset):
    """Flexible dataset that loads from numpy file paths or wraps existing tensors.

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
        super().__init__(features, targets)
        features_np = load(str(features))
        self.features = torch.from_numpy(features_np).type(dtype)
        if targets is not None:
            targets_np = load(str(targets))
            self.targets = torch.from_numpy(targets_np).type(dtype)
        else:
            self.targets = self.features

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]

    @property
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        return Shape(
            features=self.features.shape[1:],
            targets=self.targets.shape[1:],
        )
