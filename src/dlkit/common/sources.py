"""Array source protocol and open-reader result type for the DLKit data layer.

Defines the ``ArraySource`` protocol that unifies eager and lazy per-sample
array access, and the ``OpenReaderResult`` sum type returned by
``PathBasedEntry.open_reader()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class ArraySource(Protocol):
    """Protocol for per-sample array access — common across eager and lazy sources.

    Implementations must expose ``n_samples``, ``get_item``, and ``get_batch``
    so that dataset and loader code can treat zarr, npy, and in-memory sources
    uniformly.
    """

    @property
    def n_samples(self) -> int:
        """Total number of samples.

        Returns:
            Integer count of samples available in this source.
        """
        ...

    def get_item(self, idx: int) -> torch.Tensor:
        """Return a single sample tensor for the given index.

        Args:
            idx: Zero-based sample index.

        Returns:
            Tensor of shape ``(*sample_shape,)``.
        """
        ...

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a batch tensor for the given list of indices.

        Args:
            indices: List of zero-based sample indices.

        Returns:
            Tensor of shape ``(B, *sample_shape)`` where ``B = len(indices)``.
        """
        ...


# Sum type for open_reader() return: either a lazy ArraySource or a file Path.
type OpenReaderResult = ArraySource | Path
"""Return type of ``PathBasedEntry.open_reader()``."""


__all__ = [
    "ArraySource",
    "OpenReaderResult",
]
