"""Zarr IO abstractions — `ILazyReader` protocol and `ZarrLazyReader` implementation.

Lives at the ``infrastructure`` level (not under ``infrastructure.io``) so that
``infrastructure.config`` entry types can return ``ILazyReader`` instances
without creating a ``config→io`` import cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class ILazyReader(Protocol):
    """Protocol for per-sample lazy array readers.

    Implementations index into an on-disk array without loading the whole
    array into RAM.
    """

    def __getitem__(self, idx: int | list[int] | slice) -> torch.Tensor:
        """Read sample(s) by index.

        Args:
            idx: Single index, list of indices, or slice.

        Returns:
            Tensor of shape ``(sample_shape,)`` for int idx, or
            ``(n, *sample_shape)`` for list/slice.
        """
        ...

    @property
    def n_samples(self) -> int:
        """Total number of samples in the array."""
        ...

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (all dims after axis 0)."""
        ...


class ZarrLazyReader:
    """Lazy indexed reader for a native zarr array.

    Args:
        path: Path to the zarr array store directory.
    """

    def __init__(self, path: Path) -> None:
        import zarr

        self._arr = zarr.open_array(str(path), mode="r")

    def __getitem__(self, idx: int | list[int] | slice) -> torch.Tensor:
        """Read sample(s) by index with broadcast support for single-sample arrays.

        When the array contains exactly one sample (``n_samples == 1``), any
        index is resolved to ``0`` so the single sample is replicated across
        batch dimensions.

        Args:
            idx: Single index, list of indices, or slice.

        Returns:
            Tensor of shape ``(sample_shape,)`` for int idx, or
            ``(n, *sample_shape)`` for list/slice.
        """
        if self.n_samples == 1:
            match idx:
                case int():
                    return torch.from_numpy(np.array(self._arr[0]))
                case list():
                    single = np.array(self._arr[0])
                    return torch.from_numpy(np.stack([single] * len(idx)))
                case slice():
                    indices = list(range(*idx.indices(self.n_samples)))
                    single = np.array(self._arr[0])
                    return torch.from_numpy(np.stack([single] * max(1, len(indices))))
        return torch.from_numpy(np.array(self._arr[idx]))

    @property
    def n_samples(self) -> int:
        """Total number of samples in the array."""
        return int(self._arr.shape[0])

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (all dims after axis 0)."""
        return tuple(self._arr.shape[1:])
