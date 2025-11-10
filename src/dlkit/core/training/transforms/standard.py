import torch

from dlkit.core.training.transforms.base import Transform
from dlkit.core.training.transforms.interfaces import IFittableTransform, IInvertibleTransform


class StandardScaler(Transform, IFittableTransform, IInvertibleTransform):
    def __init__(
        self, dim: int | list[int] | None = None, input_shape: tuple[int, ...] | None = None
    ) -> None:
        super().__init__(input_shape)
        # Convert tensor to tuple for torch.zeros/ones
        shape_tuple = tuple(self.input_shape.tolist()) if isinstance(self.input_shape, torch.Tensor) else tuple(self.input_shape)
        self.register_buffer("mean", torch.zeros(shape_tuple))
        self.register_buffer("std", torch.ones(shape_tuple))
        self.dim = dim or 0

    def fit(self, data: torch.Tensor) -> None:
        self.mean = torch.mean(data, dim=self.dim, keepdim=True)
        self.std = torch.std(data, dim=self.dim, keepdim=True)
        self.fitted = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x) -> torch.Tensor:
        """Return a module that lazily computes the inverse transformation
        using the current min and max values at runtime.
        """
        return (x * self.std) + self.mean
