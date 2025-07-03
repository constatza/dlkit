import torch
from pathlib import Path
from typing import TypedDict
from .base import BaseDataset
from dlkit.datatypes.dataset import Shape
from dlkit.io import load_array
from .base import register_dataset


class SupervisedData(TypedDict):
    """Data class for supervised datasets."""

    x: torch.Tensor
    y: torch.Tensor


@register_dataset
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
        self.x = load_array(x, dtype=dtype)
        self.y = load_array(y, dtype=dtype) if y else self.x

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
