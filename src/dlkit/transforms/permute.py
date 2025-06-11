import torch

from dlkit.transforms.base import Transform

epsilon = 1e-8


class Permutation(Transform):
    def __init__(self, *, dims: tuple[int], input_shape: tuple[int, ...]):
        """Must be a permutation and must be used before any other transforms."""
        super().__init__(input_shape=input_shape)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        inverse_dims = [0] * len(self.dims)
        for i, dim in enumerate(self.dims):
            inverse_dims[dim] = i

        return y.permute(*inverse_dims)
