import torch

from dlkit.transforms.base import Transform


class StandardScaler(Transform):
    mean: torch.Tensor | None = None

    def __init__(
        self, dim: int | list[int] | None = None, input_shape: tuple[int, ...] | None = None
    ) -> None:
        super().__init__()
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.input_shape = (1, *input_shape)
        self.register_buffer("mean", torch.zeros(self.input_shape))
        self.register_buffer("std", torch.ones(self.input_shape))
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
