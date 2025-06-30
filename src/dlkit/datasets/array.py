from typing import Protocol
from pathlib import Path

import torch
from dlkit.io import load_array
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
        self.features = load_array(features, dtype=dtype)
        if targets is not None:
            self.targets = load_array(targets, dtype=dtype)
        else:
            self.targets = self.features

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
        }

    @property
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        return Shape(
            features=self.features.shape[1:],
            targets=self.targets.shape[1:],
        )
