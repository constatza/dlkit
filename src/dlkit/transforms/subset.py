import torch
from collections.abc import Sequence
from pydantic import validate_call, ConfigDict
from .base import Transform
from dlkit.utils.general import slice_to_list


class TensorSubset(Transform):
    """
    Subsample a tensor along a given dimension.
    """

    _keep: Sequence[int] | slice

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        keep: Sequence[int] | slice,
        dim: int = 1,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        """
        Args:
            keep (Sequence[int] | slice): Indices to keep along `dim`.
            dim (int, optional): Dimension along which to subset. Defaults to 1.
            input_shape (tuple[int, ...], optional): The shape of the input data.
                Defaults to None.
        """
        super().__init__()
        self.dim: int = dim
        self.length = None
        self._keep = keep
        if input_shape:
            self.fit(torch.zeros(input_shape))

    @property
    def keep(self):
        if isinstance(self._keep, slice):
            return slice_to_list(self._keep, self.length)
        return self._keep

    def fit(self, data: torch.Tensor) -> None:
        self.length = data.size(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of arbitrary shape.
        Returns:
            A tensor where, along `self.dim`, only indices in `self.keep`
            that are not in `self.drop` are retained, in ascending order.
        """

        # Build a sorted list of indices in [0, size_along_dim)
        # excluding those in drop and keeping only those in keep
        self.fit(x)
        final_indices = sorted(set(range(self.length)) & set(self.keep))

        # Create a full‐dimensional “slice(None)” list
        indexer = [slice(None)] * x.ndim
        indexer[self.dim] = list(final_indices)  # use list of ints to index that dim
        return x[tuple(indexer)]

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x
