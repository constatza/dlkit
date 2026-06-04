"""Protocols and enums for dense array pack I/O."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, Self, runtime_checkable

import numpy as np
import torch
from torch import Tensor


class ArrayPackFormat(StrEnum):
    """Supported dense array storage formats.

    Used as the registry key in ``register_format`` and as the ``format=``
    argument to ``open_array_pack``, ``write_array_pack``, and
    ``save_array_pack``.

    New formats can be registered via ``register_format()`` — no enum change
    is required for third-party backends.
    """

    ZARR_DENSE = "zarr_dense"
    """Zarr v3 chunked dense arrays. No pre-allocation; the library default."""
    # Future: ZARR_CSR = "zarr_csr", HDF5_DENSE = "hdf5_dense"


@runtime_checkable
class IArrayPackReader(Protocol):
    """Read protocol for multi-sample matrix packs (structural typing).

    Each matrix in the pack has identical shape ``(rows, cols)``. Indexing
    always returns dense ``Tensor`` objects.
    """

    @property
    def n_samples(self) -> int:
        """Number of matrices in the pack."""
        ...

    @property
    def matrix_size(self) -> tuple[int, int]:
        """Shared ``(rows, cols)`` shape for every sample."""
        ...

    def __getitem__(self, idx: int | list[int] | slice) -> Tensor:
        """Return one or more dense matrices.

        Args:
            idx: ``int`` → ``Tensor[rows, cols]``;
                 ``list[int]`` or ``slice`` → ``Tensor[B, rows, cols]``.

        Returns:
            Dense ``Tensor`` shaped ``(rows, cols)`` for scalar index, or
            ``(B, rows, cols)`` for list/slice index.
        """
        ...


@runtime_checkable
class IArrayPackWriter(Protocol):
    """Streaming write protocol for array packs.

    Implementations must be usable as context managers. On ``__exit__``
    (or explicit ``close()``) the on-disk pack is finalised and safe to read.
    """

    def write_sample(self, data: np.ndarray) -> None:
        """Append one ``(rows, cols)`` matrix to the pack.

        Args:
            data: Dense numpy array of shape ``(rows, cols)``.  Scipy sparse
                matrices are accepted if scipy is installed.
        """
        ...

    def write_samples(self, data: np.ndarray) -> None:
        """Append a batch of ``(K, rows, cols)`` matrices to the pack.

        Args:
            data: Dense numpy array of shape ``(K, rows, cols)``.  Scipy
                sparse matrices in batched form are accepted when scipy is
                installed.
        """
        ...

    def close(self) -> None:
        """Finalise and flush the pack. Idempotent."""
        ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: Any) -> None: ...


class AbstractArrayPackReader(ABC):
    """LSP-compliant base class for all array pack readers.

    Concrete subclasses must implement ``__getitem__``, ``n_samples``, and
    ``matrix_size``.  The deprecated ``collect`` / ``collect_stacked`` helpers
    delegate to ``__getitem__`` so existing call-sites continue to work without
    re-implementing the logic.

    Postconditions:
        - ``__getitem__(int)`` returns a dense ``Tensor`` with shape
          ``(rows, cols)``.
        - ``__getitem__(list)`` / ``__getitem__(slice)`` returns a dense
          ``Tensor`` with shape ``(B, rows, cols)``.
        - Any two ``AbstractArrayPackReader`` subclasses are substitutable
          (LSP).
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of matrices in the pack."""

    @property
    @abstractmethod
    def matrix_size(self) -> tuple[int, int]:
        """Shared ``(rows, cols)`` shape for every sample."""

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Alias for ``matrix_size`` generalizing ND arrays.

        Returns:
            The shape of each sample, as a tuple of ints.
        """
        return tuple(self.matrix_size)

    @abstractmethod
    def __getitem__(self, idx: int | list[int] | slice) -> Tensor:
        """Return one or more dense matrices.

        Args:
            idx: ``int`` → ``Tensor[rows, cols]``;
                 ``list[int]`` or ``slice`` → ``Tensor[B, rows, cols]``.

        Returns:
            Dense ``Tensor``.
        """

    # ------------------------------------------------------------------
    # Deprecated compatibility shims
    # ------------------------------------------------------------------

    def collect(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return one dense matrix. Deprecated — use ``reader[i]`` instead.

        Args:
            sample_index: Index of the sample.
            device: Move result to this device when not ``None``.
            dtype: Cast result to this dtype when not ``None``.

        Returns:
            Dense ``Tensor`` of shape ``(rows, cols)``.
        """
        warnings.warn(
            "collect() is deprecated; use reader[i] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        t = self[sample_index]
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t

    def collect_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return a stacked batch tensor. Deprecated — use ``reader[list]`` instead.

        Args:
            sample_indices: Indices of the samples to stack.
            device: Move result to this device when not ``None``.
            dtype: Cast result to this dtype when not ``None``.

        Returns:
            Dense ``Tensor`` of shape ``(B, rows, cols)``.
        """
        warnings.warn(
            "collect_stacked() is deprecated; use reader[indices] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        t = self[list(sample_indices)]
        if dtype is not None:
            t = t.to(dtype=dtype)
        if device is not None:
            t = t.to(device=device)
        return t
