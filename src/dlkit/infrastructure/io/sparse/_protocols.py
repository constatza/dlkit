"""Protocols and enums for sparse pack I/O."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from ._manifest import PackFiles, PackManifest


class SparseFormat(StrEnum):
    """Supported sparse storage formats."""

    COO = "coo"
    CSR = "csr"


@runtime_checkable
class SparseWriter(Protocol):
    """Protocol for sparse format writers (ISP: write concern only)."""

    def save(
        self,
        path: Path,
        indices: np.ndarray,
        values: np.ndarray,
        nnz_ptr: np.ndarray,
        size: tuple[int, int],
        *,
        dtype: np.dtype | None = None,
        manifest: PackManifest | None = None,
        files: PackFiles | None = None,
    ) -> None:
        """Persist sparse pack arrays to disk."""
        ...


@runtime_checkable
class SparseLoader(Protocol):
    """Protocol for sparse format loaders (ISP: read concern only)."""

    def load_arrays(
        self,
        path: Path,
        files: PackFiles | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load sparse pack arrays (indices, values, nnz_ptr) from disk."""
        ...

    def load_size(
        self,
        path: Path,
        files: PackFiles | None = None,
    ) -> tuple[int, int]:
        """Load matrix size (rows, cols) from disk."""
        ...


@runtime_checkable
class SparseCodec(SparseWriter, SparseLoader, Protocol):
    """Full codec: both write and load capabilities."""


@runtime_checkable
class SparsePackReader(Protocol):
    """Format-agnostic sparse pack reader protocol.

    NOTE: keep in sync with AbstractSparsePackReader — both define the same interface,
    one for structural typing, one for nominal enforcement.
    """

    @property
    def n_samples(self) -> int:
        """Number of sparse matrices in the pack."""
        ...

    @property
    def matrix_size(self) -> tuple[int, int]:
        """Matrix shape for each sample."""
        ...

    def build_torch_sparse(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build one sparse tensor for a sample index."""
        ...

    def build_torch_sparse_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build a stacked sparse tensor with shape (B, rows, cols)."""
        ...


class AbstractSparsePackReader(ABC):
    """ABC for LSP enforcement on concrete sparse pack readers.

    All concrete readers must inherit this class, ensuring all interface
    methods are implemented at class definition time.

    NOTE: keep in sync with SparsePackReader protocol — both define the same interface.

    Postconditions:
        - ``build_torch_sparse`` returns an is-sparse Tensor.
        - ``build_torch_sparse_stacked`` returns an is-sparse Tensor with shape (B, rows, cols).
        - Stored packs are always coalesced (sorted unique coordinates enforced at write time).
        - Any ``AbstractSparsePackReader`` subclass is substitutable for any other (LSP).
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of sparse matrices in the pack."""

    @property
    @abstractmethod
    def matrix_size(self) -> tuple[int, int]:
        """Matrix shape for each sample."""

    @abstractmethod
    def build_torch_sparse(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build one sparse tensor for a sample index.

        Postcondition: returned tensor must have ``is_sparse == True`` and be coalesced.
        """

    @abstractmethod
    def build_torch_sparse_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Build a stacked sparse tensor for many sample indices.

        Postcondition: returned tensor must have ``is_sparse == True`` and shape (B, rows, cols).
        """
