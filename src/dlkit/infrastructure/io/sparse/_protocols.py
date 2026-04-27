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
        value_scale: float = 1.0,
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
        """Load sparse pack arrays from disk."""
        ...

    def load_value_scale(
        self,
        path: Path,
        files: PackFiles | None = None,
    ) -> float:
        """Load sparse pack value scale from disk (legacy default: 1.0)."""
        ...


@runtime_checkable
class SparseCodec(SparseWriter, SparseLoader, Protocol):
    """Full codec: both write and load capabilities."""


@runtime_checkable
class SparsePackReader(Protocol):
    """Format-agnostic sparse pack reader protocol."""

    @property
    def n_samples(self) -> int:
        """Number of sparse matrices in the pack."""
        ...

    @property
    def matrix_size(self) -> tuple[int, int]:
        """Matrix shape for each sample."""
        ...

    @property
    def value_scale(self) -> float:
        """Value scale for stored sparse values."""
        ...

    def build_torch_sparse(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build one sparse tensor for a sample index."""
        ...

    def build_torch_sparse_batch(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> list[Tensor]:
        """Build sparse tensors for many sample indices."""
        ...

    def build_torch_sparse_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build a stacked sparse tensor with shape (B, rows, cols)."""
        ...


class AbstractSparsePackReader(ABC):
    """ABC for LSP enforcement on concrete sparse pack readers.

    All concrete readers must inherit this class, ensuring all interface
    methods are implemented at class definition time.

    Postconditions:
        - ``build_torch_sparse`` must return an is-sparse Tensor.
        - ``build_torch_sparse_batch`` must return a list of is-sparse Tensors.
        - Any ``AbstractSparsePackReader`` subclass is substitutable for any
          other (Liskov Substitution Principle).
    """

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of sparse matrices in the pack."""

    @property
    @abstractmethod
    def matrix_size(self) -> tuple[int, int]:
        """Matrix shape for each sample."""

    @property
    @abstractmethod
    def value_scale(self) -> float:
        """Value scale for stored sparse values."""

    @abstractmethod
    def build_torch_sparse(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build one sparse tensor for a sample index.

        Postcondition: returned tensor must have ``is_sparse == True``.
        """

    @abstractmethod
    def build_torch_sparse_batch(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> list[Tensor]:
        """Build sparse tensors for many sample indices.

        Postcondition: each returned tensor must have ``is_sparse == True``.
        """

    @abstractmethod
    def build_torch_sparse_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build a stacked sparse tensor for many sample indices.

        Postcondition: returned tensor must have ``is_sparse == True``.
        """
