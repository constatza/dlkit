from collections.abc import Sequence
from typing import cast

import torch
from pydantic import ConfigDict, validate_call

from dlkit.tools.utils.general import slice_to_list

from .base import Transform


class TensorSubset(Transform):
    """Subsample a tensor along a given dimension (shape-agnostic transform).

    This transform selects specific indices along a dimension. While it can
    benefit from knowing the dimension size (via fit()), it works shape-agnostically
    by inferring the dimension size from data when needed.

    Example:
        >>> # Keep specific indices along dimension 1
        >>> subset = TensorSubset(keep=[0, 2, 5], dim=1)
        >>> data = torch.randn(32, 100)  # batch_size=32, features=100
        >>> subsetted = subset(data)  # Shape: (32, 3)
    """

    _keep: Sequence[int] | slice
    dim: int
    length: int | None

    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def __init__(
        self,
        *,
        keep: Sequence[int] | slice,
        dim: int = 1,
    ) -> None:
        """Initialize tensor subset transform.

        Args:
            keep: Indices to keep along the specified dimension.
                Can be a sequence of ints or a slice object.
            dim: Dimension along which to subset. Defaults to 1.

        Example:
            >>> # Keep first 10 features
            >>> subset = TensorSubset(keep=slice(0, 10), dim=1)
            >>>
            >>> # Keep specific indices
            >>> subset = TensorSubset(keep=[0, 5, 10, 15], dim=1)
        """
        super().__init__()
        self.dim = dim
        self.length = None
        self._keep = keep

    @property
    def keep(self) -> Sequence[int]:
        if isinstance(self._keep, slice):
            assert self.length is not None, "length must be set before accessing keep as slice"
            return slice_to_list(self._keep, self.length)
        return self._keep

    def fit(self, data: torch.Tensor) -> None:
        self.length = data.size(self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Subset tensor along specified dimension.

        Args:
            x: Input tensor of arbitrary shape.

        Returns:
            Tensor with only specified indices kept along self.dim.
            Shape is same as input except dimension self.dim is reduced.

        Example:
            >>> subset = TensorSubset(keep=[0, 2, 5], dim=1)
            >>> data = torch.randn(32, 100)
            >>> result = subset(data)  # Shape: (32, 3)
        """
        # Auto-fit from data if not already fitted
        if self.length is None:
            self.fit(x)
        assert self.length is not None  # guaranteed by fit()

        # Build sorted list of indices to keep
        final_indices = sorted(set(range(self.length)) & set(self.keep))

        # Create full-dimensional slice(None) list; element type is slice | list[int]
        indexer = cast(list[slice | list[int]], [slice(None)] * x.ndim)
        indexer[self.dim] = list(final_indices)  # Index specific dimension
        return x[tuple(indexer)]

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Identity inverse transform (no true inverse for subset).

        Args:
            x: Subsetted tensor.

        Returns:
            Same tensor (no reconstruction of dropped indices).
        """
        return x

    def infer_output_shape(self, in_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Infer output shape. TensorSubset reduces the specified dimension.

        Args:
            in_shape: Input tensor shape.

        Returns:
            Output shape with dimension dim = len(keep).
        """
        output_shape = list(in_shape)

        # Compute length from keep parameter
        if isinstance(self._keep, slice):
            # Handle slice objects
            start = self._keep.start or 0
            stop = self._keep.stop or in_shape[self.dim]
            step = self._keep.step or 1
            length = len(range(start, stop, step))
        else:
            # Handle lists/tuples
            length = len(self._keep)

        output_shape[self.dim] = length
        return tuple(output_shape)
