"""COO sparse pack codec and reader implementation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ._manifest import (
    PackFiles,
    PackManifest,
    _manifest_from_arrays,
    _normalize_value_scale,
)
from ._protocols import AbstractSparsePackReader, SparseCodec, SparseFormat, SparseLoader


def _torch_dtype_from_numpy_name(dtype_name: str) -> torch.dtype:
    """Convert a numpy dtype name to the corresponding torch dtype.

    Args:
        dtype_name: A numpy dtype string (e.g. ``"float32"``).

    Returns:
        The corresponding ``torch.dtype``.
    """
    numpy_dtype = np.dtype(dtype_name)
    return torch.from_numpy(np.empty((), dtype=numpy_dtype)).dtype


def _coalesce_sample_payload(
    sample_indices: np.ndarray,
    sample_values: np.ndarray,
    *,
    n_cols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return canonical COO payload for one sample (sorted, unique coordinates)."""
    if sample_values.size == 0:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=sample_values.dtype)

    row_idx = sample_indices[0]
    col_idx = sample_indices[1]
    linear_idx = row_idx * n_cols + col_idx
    order = np.argsort(linear_idx, kind="mergesort")
    linear_sorted = linear_idx[order]
    values_sorted = sample_values[order]

    is_unique_start = np.empty(linear_sorted.size, dtype=bool)
    is_unique_start[0] = True
    is_unique_start[1:] = linear_sorted[1:] != linear_sorted[:-1]

    unique_linear = linear_sorted[is_unique_start]
    reduce_starts = np.flatnonzero(is_unique_start)
    reduced_values = np.add.reduceat(values_sorted, reduce_starts).astype(
        sample_values.dtype, copy=False
    )

    rows = (unique_linear // n_cols).astype(np.int64, copy=False)
    cols = (unique_linear % n_cols).astype(np.int64, copy=False)
    reduced_indices = np.vstack([rows, cols]).astype(np.int64, copy=False)
    return reduced_indices, reduced_values


def _coalesce_pack_payload(
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    *,
    matrix_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonicalize each sample payload once at write time."""
    n_cols = int(matrix_size[1])
    coalesced_index_parts: list[np.ndarray] = []
    coalesced_value_parts: list[np.ndarray] = []
    ptr_out = [0]

    for sample_idx in range(int(nnz_ptr.size - 1)):
        start = int(nnz_ptr[sample_idx])
        end = int(nnz_ptr[sample_idx + 1])
        sample_indices = indices[:, start:end]
        sample_values = values[start:end]
        reduced_indices, reduced_values = _coalesce_sample_payload(
            sample_indices, sample_values, n_cols=n_cols
        )
        coalesced_index_parts.append(reduced_indices)
        coalesced_value_parts.append(reduced_values)
        ptr_out.append(ptr_out[-1] + int(reduced_values.size))

    if coalesced_index_parts:
        out_indices = np.concatenate(coalesced_index_parts, axis=1)
    else:
        out_indices = np.empty((2, 0), dtype=np.int64)

    if coalesced_value_parts:
        out_values = np.concatenate(coalesced_value_parts, axis=0).astype(values.dtype, copy=False)
    else:
        out_values = np.empty((0,), dtype=values.dtype)

    out_ptr = np.asarray(ptr_out, dtype=np.int64)
    return out_indices, out_values, out_ptr


def _value_scale_from_array(raw_scale: np.ndarray) -> float:
    """Convert loaded numpy scale payload to a validated scalar float.

    Args:
        raw_scale: Numpy array that is either 0-D or shape (1,).

    Returns:
        Validated scale as a Python float.

    Raises:
        ValueError: If the array has an unexpected shape or an invalid scale value.
    """
    if raw_scale.ndim == 0:
        scale = float(raw_scale)
    elif raw_scale.ndim == 1 and raw_scale.size == 1:
        scale = float(raw_scale[0])
    else:
        raise ValueError(
            f"values_scale payload must be a scalar or shape (1,), got {raw_scale.shape}"
        )
    return _normalize_value_scale(scale)


class CooPackCodec(SparseCodec):
    """Writer/loader codec for COO sparse packs.

    Implements both ``SparseWriter`` and ``SparseLoader``.  All array validation
    and manifest consistency checks run *before* any filesystem writes, so a
    validation failure never leaves a partially-written pack on disk.
    """

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
        """Save COO sparse pack arrays to disk.

        Validation order:
        1. Coerce arrays (pure).
        2. Validate array shapes and pointer consistency (pure — raises before I/O).
        3. Validate manifest consistency if provided (pure — raises before I/O).
        4. Create directory and write files (side effects only after full validation).

        Args:
            path: Destination directory for the sparse pack.
            indices: COO indices array of shape (2, total_nnz).
            values: COO values array of shape (total_nnz,).
            nnz_ptr: Row pointer array of shape (n_samples + 1,).
            size: Explicit matrix dimensions (rows, cols).
            dtype: Force a specific numpy dtype for stored values.
            value_scale: Normalization scale; must be finite and > 0.
            manifest: Authoritative contract; used instead of ``size``/``dtype``/
                ``value_scale`` when provided.
            files: Custom payload filenames; ignored if ``manifest`` is provided.

        Raises:
            ValueError: On any shape mismatch, invalid pointer, or manifest
                inconsistency — always before any filesystem write.
        """
        payload_files = manifest.files if manifest is not None else (files or PackFiles())

        # 1. Coerce arrays (pure)
        indices_arr = np.asarray(indices, dtype=np.int64)
        if manifest is not None:
            values_dtype = np.dtype(manifest.dtype)
            resolved_scale = _normalize_value_scale(manifest.value_scale)
            expected_size = (int(manifest.matrix_size[0]), int(manifest.matrix_size[1]))
        else:
            values_dtype = np.dtype(dtype) if dtype is not None else np.asarray(values).dtype
            resolved_scale = _normalize_value_scale(value_scale)
            expected_size = (int(size[0]), int(size[1]))
        values_arr = np.asarray(values, dtype=values_dtype)
        nnz_ptr_arr = np.asarray(nnz_ptr, dtype=np.int64)

        # 2. Validate arrays (pure — raises before any I/O)
        if indices_arr.ndim != 2 or indices_arr.shape[0] != 2:
            raise ValueError(f"indices must have shape (2, total_nnz), got {indices_arr.shape}")
        if values_arr.ndim != 1:
            raise ValueError(f"values must be 1D, got {values_arr.shape}")
        if nnz_ptr_arr.ndim != 1:
            raise ValueError(f"nnz_ptr must be 1D, got {nnz_ptr_arr.shape}")
        if len(size) != 2:
            raise ValueError(f"size must be 2D, got {size}")
        if tuple(int(v) for v in size) != expected_size:
            raise ValueError(
                f"size {tuple(int(v) for v in size)} does not match manifest matrix_size {expected_size}"
            )
        if nnz_ptr_arr.size < 2:
            raise ValueError(f"nnz_ptr must include at least start/end, got {nnz_ptr_arr.size}")
        if indices_arr.shape[1] != values_arr.size:
            raise ValueError(
                f"indices nnz ({indices_arr.shape[1]}) does not match values ({values_arr.size})"
            )
        if nnz_ptr_arr[0] != 0:
            raise ValueError("nnz_ptr must start at 0")
        if nnz_ptr_arr[-1] != values_arr.size:
            raise ValueError(
                f"nnz_ptr last value ({nnz_ptr_arr[-1]}) must equal total nnz ({values_arr.size})"
            )
        if np.any(np.diff(nnz_ptr_arr) < 0):
            raise ValueError("nnz_ptr must be non-decreasing")
        row_idx = indices_arr[0]
        col_idx = indices_arr[1]
        rows, cols = expected_size
        if np.any(row_idx < 0) or np.any(row_idx >= rows):
            raise ValueError(f"row indices must be in [0, {rows}), got out-of-bounds entries")
        if np.any(col_idx < 0) or np.any(col_idx >= cols):
            raise ValueError(f"column indices must be in [0, {cols}), got out-of-bounds entries")

        # Canonicalize once at write-time: sorted unique coordinates per sample.
        indices_arr, values_arr, nnz_ptr_arr = _coalesce_pack_payload(
            indices_arr,
            values_arr,
            nnz_ptr_arr,
            matrix_size=expected_size,
        )

        # 3. Validate manifest consistency (pure — raises before any I/O)
        if manifest is not None:
            expected_n_samples = int(nnz_ptr_arr.size - 1)
            expected_total_nnz = int(values_arr.size)
            if manifest.format != SparseFormat.COO:
                raise ValueError(
                    f"CooPackCodec requires COO manifest, got '{manifest.format.value}'"
                )
            if manifest.n_samples != expected_n_samples:
                raise ValueError(
                    f"manifest n_samples ({manifest.n_samples}) does not match payload ({expected_n_samples})"
                )
            if manifest.total_nnz != expected_total_nnz:
                raise ValueError(
                    f"manifest total_nnz ({manifest.total_nnz}) does not match payload ({expected_total_nnz})"
                )

        # 4. Side effects — mkdir and file writes
        pack_dir = Path(path)
        pack_dir.mkdir(parents=True, exist_ok=True)
        np.save(pack_dir / payload_files.indices, indices_arr)
        np.save(pack_dir / payload_files.values, values_arr)
        np.save(pack_dir / payload_files.nnz_ptr, nnz_ptr_arr)
        np.save(
            pack_dir / payload_files.values_scale,
            np.asarray(resolved_scale, dtype=np.float64),
        )

    def load_arrays(
        self,
        path: Path,
        files: PackFiles | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load COO sparse pack arrays from disk.

        Args:
            path: Pack directory.
            files: Custom payload filenames; defaults to ``PackFiles()``.

        Returns:
            Tuple of ``(indices, values, nnz_ptr)`` numpy arrays.
        """
        pack_dir = Path(path)
        payload_files = files or PackFiles()
        indices = np.load(pack_dir / payload_files.indices, allow_pickle=False)
        values = np.load(pack_dir / payload_files.values, allow_pickle=False)
        nnz_ptr = np.load(pack_dir / payload_files.nnz_ptr, allow_pickle=False)
        return indices, values, nnz_ptr

    def load_value_scale(
        self,
        path: Path,
        files: PackFiles | None = None,
    ) -> float:
        """Load value scale payload; returns 1.0 for legacy packs without the file.

        Args:
            path: Pack directory.
            files: Custom payload filenames; defaults to ``PackFiles()``.

        Returns:
            Validated scale float.
        """
        pack_dir = Path(path)
        payload_files = files or PackFiles()
        scale_path = pack_dir / payload_files.values_scale
        if not scale_path.exists():
            return 1.0
        raw_scale = np.load(scale_path, allow_pickle=False)
        return _value_scale_from_array(np.asarray(raw_scale))


class CooPackReader(AbstractSparsePackReader):
    """Reader for COO sparse packs with direct torch sparse tensor builders.

    Implements ``AbstractSparsePackReader`` for LSP compliance.  Accepts an
    injected ``SparseLoader`` for testability (default: ``CooPackCodec()``).
    """

    def __init__(
        self,
        path: Path,
        manifest: PackManifest | None = None,
        *,
        files: PackFiles | None = None,
        matrix_size: tuple[int, int] | None = None,
        dtype: np.dtype | str | None = None,
        loader: SparseLoader | None = None,
    ) -> None:
        """Initialise the reader and load all arrays from disk.

        Args:
            path: Pack directory.
            manifest: Authoritative contract; validated against payload if provided.
            files: Custom payload filenames; used only when ``manifest`` is None.
            matrix_size: Explicit matrix dimensions for manifest inference.
            dtype: Explicit dtype for manifest inference.
            loader: ``SparseLoader`` implementation; defaults to ``CooPackCodec()``.

        Raises:
            ValueError: If the manifest is not for COO format or does not match the payload.
        """
        self._path = Path(path)
        self._loader: SparseLoader = loader if loader is not None else CooPackCodec()

        if manifest is not None:
            if manifest.format != SparseFormat.COO:
                raise ValueError(
                    f"CooPackReader requires COO manifest, got '{manifest.format.value}'"
                )
            self._manifest = manifest
            self._indices, self._values, self._nnz_ptr = self._loader.load_arrays(
                self._path,
                files=manifest.files,
            )
            self._value_scale = _normalize_value_scale(manifest.value_scale)
            if int(self._nnz_ptr.size - 1) != manifest.n_samples:
                raise ValueError(
                    f"manifest n_samples ({manifest.n_samples}) does not match payload "
                    f"({int(self._nnz_ptr.size - 1)})"
                )
            if int(self._values.size) != manifest.total_nnz:
                raise ValueError(
                    f"manifest total_nnz ({manifest.total_nnz}) does not match payload "
                    f"({int(self._values.size)})"
                )
        else:
            resolved_files = files or PackFiles()
            self._indices, self._values, self._nnz_ptr = self._loader.load_arrays(
                self._path,
                files=resolved_files,
            )
            self._value_scale = self._loader.load_value_scale(self._path, files=resolved_files)
            self._manifest = _manifest_from_arrays(
                indices=self._indices,
                values=self._values,
                nnz_ptr=self._nnz_ptr,
                files=resolved_files,
                matrix_size=matrix_size,
                dtype=dtype,
                value_scale=self._value_scale,
            )

        self._default_torch_dtype = _torch_dtype_from_numpy_name(self._manifest.dtype)

    @property
    def n_samples(self) -> int:
        """Number of sparse matrices in the pack."""
        return self._manifest.n_samples

    @property
    def matrix_size(self) -> tuple[int, int]:
        """Matrix shape for each sample."""
        return self._manifest.matrix_size

    @property
    def value_scale(self) -> float:
        """Value scale for stored sparse values."""
        return self._value_scale

    def _resolve_sample_index(self, sample_index: int) -> int:
        """Resolve sample index with shared-matrix broadcast semantics.

        A pack with ``n_samples == 1`` broadcasts that single matrix to any index.

        Args:
            sample_index: Non-negative sample index.

        Returns:
            Resolved index (0 for broadcast packs).

        Raises:
            IndexError: If index is negative or out of range.
        """
        if sample_index < 0:
            raise IndexError(f"sample_index must be >= 0, got {sample_index}")
        if self.n_samples == 1:
            return 0
        if sample_index >= self.n_samples:
            raise IndexError(
                f"sample_index {sample_index} out of range for {self.n_samples} samples"
            )
        return sample_index

    def _resolve_sample_indices(self, sample_indices: Sequence[int]) -> np.ndarray:
        """Resolve many sample indices with shared-matrix broadcast semantics."""
        resolved = np.asarray(sample_indices, dtype=np.int64)
        if resolved.ndim != 1:
            raise ValueError(f"sample_indices must be 1-D, got shape {resolved.shape}")
        if resolved.size == 0:
            return resolved
        if np.any(resolved < 0):
            negative = int(resolved[resolved < 0][0])
            raise IndexError(f"sample_index must be >= 0, got {negative}")
        if self.n_samples == 1:
            return np.zeros_like(resolved, dtype=np.int64)
        max_index = int(resolved.max())
        if max_index >= self.n_samples:
            raise IndexError(f"sample_index {max_index} out of range for {self.n_samples} samples")
        return resolved

    def build_torch_sparse(
        self,
        sample_index: int,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build one sparse COO tensor for a sample index.

        Args:
            sample_index: Index of the sample to build.
            device: Target device.
            dtype: Target dtype; defaults to the pack's stored dtype.
            denormalize: If True, multiply values by ``value_scale``.
            coalesce: If True, call ``coalesce()`` on the result.

        Returns:
            Sparse COO tensor with ``is_sparse == True``.
        """
        resolved_index = self._resolve_sample_index(sample_index)
        start = int(self._nnz_ptr[resolved_index])
        end = int(self._nnz_ptr[resolved_index + 1])

        indices = torch.from_numpy(self._indices[:, start:end])
        values = torch.from_numpy(self._values[start:end])
        target_dtype = dtype or self._default_torch_dtype
        if values.dtype != target_dtype:
            values = values.to(dtype=target_dtype)
        if denormalize:
            values = values * self._value_scale

        sparse = torch.sparse_coo_tensor(
            indices,
            values,
            size=self.matrix_size,
            device=device,
            is_coalesced=True,
        )
        return sparse.coalesce() if coalesce else sparse

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

        Args:
            sample_indices: Indices of the samples to build.
            device: Target device.
            dtype: Target dtype; defaults to the pack's stored dtype.
            denormalize: If True, multiply values by ``value_scale``.
            coalesce: If True, call ``coalesce()`` on each result.

        Returns:
            List of sparse COO tensors, each with ``is_sparse == True``.
        """
        return [
            self.build_torch_sparse(
                sample_index=sample_index,
                device=device,
                dtype=dtype,
                denormalize=denormalize,
                coalesce=coalesce,
            )
            for sample_index in sample_indices
        ]

    def build_torch_sparse_stacked(
        self,
        sample_indices: Sequence[int],
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        denormalize: bool = False,
        coalesce: bool = False,
    ) -> Tensor:
        """Build one stacked sparse COO tensor with shape (B, rows, cols)."""
        batch_size = len(sample_indices)
        rows, cols = self.matrix_size
        if batch_size == 0:
            empty_indices = torch.empty((3, 0), dtype=torch.int64)
            empty_values = torch.empty((0,), dtype=dtype or self._default_torch_dtype)
            sparse = torch.sparse_coo_tensor(
                empty_indices,
                empty_values,
                size=(0, rows, cols),
                device=device,
                is_coalesced=True,
            )
            return sparse.coalesce() if coalesce else sparse

        resolved_indices = self._resolve_sample_indices(sample_indices)
        starts = self._nnz_ptr[resolved_indices]
        ends = self._nnz_ptr[resolved_indices + 1]
        nnz_lengths = ends - starts
        total_nnz = int(nnz_lengths.sum())

        out_indices_np = np.empty((3, total_nnz), dtype=np.int64)
        out_values_np = np.empty((total_nnz,), dtype=self._values.dtype)
        out_indices_np[0] = np.repeat(np.arange(batch_size, dtype=np.int64), nnz_lengths)

        cursor = 0
        for sample_start, sample_end, nnz_len in zip(starts, ends, nnz_lengths, strict=True):
            if nnz_len == 0:
                continue
            sample_start_int = int(sample_start)
            sample_end_int = int(sample_end)
            nnz_len_int = int(nnz_len)
            out_indices_np[1:3, cursor : cursor + nnz_len_int] = self._indices[
                :, sample_start_int:sample_end_int
            ]
            out_values_np[cursor : cursor + nnz_len_int] = self._values[
                sample_start_int:sample_end_int
            ]
            cursor += nnz_len_int

        out_indices = torch.from_numpy(out_indices_np)
        out_values = torch.from_numpy(out_values_np)
        target_dtype = dtype or self._default_torch_dtype
        if out_values.dtype != target_dtype:
            out_values = out_values.to(dtype=target_dtype)
        if denormalize:
            out_values = out_values * self._value_scale

        sparse = torch.sparse_coo_tensor(
            out_indices,
            out_values,
            size=(batch_size, rows, cols),
            device=device,
            is_coalesced=True,
        )
        return sparse.coalesce() if coalesce else sparse


# Re-export for backward compatibility with code that referenced the old name.
CooPackWriter = CooPackCodec
