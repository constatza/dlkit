from pathlib import Path

import torch
from dlkit.datatypes.dataset import Shape
from .base import BaseDataset
from typing import TypedDict


class SupervisedData(TypedDict):
    """Data class for supervised datasets."""

    x: torch.Tensor
    y: torch.Tensor


class SupervisedArrayDataset(BaseDataset):
    """Flexible dataset that loads from numpy file paths or wraps existing tensors.

    Args:
        x: Path to numpy file or a torch.Tensor of x.
        y: Path to numpy file or a torch.Tensor of targets.
    """

    def __init__(
        self,
        x: Path,
        y: Path | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(x, y or x)
        self.x = self.tensors[0]
        self.y = self.tensors[1]

    def __getitem__(self, idx: int) -> SupervisedData:
        x, y = super().__getitem__(idx)
        return SupervisedData(x=x, y=y)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the dataset."""
        return Shape(
            features=self.x.shape[1:],
            targets=self.y.shape[1:],
        )
