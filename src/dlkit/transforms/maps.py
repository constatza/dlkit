import torch

from dlkit.transforms.base import Map

epsilon = 1e-8


class Permutation(Map):
    def __init__(self, dims: tuple[int]):
        """Must be a permutation and must be used before any other transforms"""
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self.dims)

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        inverse_dims = [0] * len(self.dims)
        for i, dim in enumerate(self.dims):
            inverse_dims[dim] = i

        return y.permute(*inverse_dims)
