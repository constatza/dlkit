"""Format-agnostic sparse pack factory functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._manifest import PackFiles, PackManifest
from ._protocols import AbstractSparsePackReader, SparseFormat
from ._registry import get_codec, get_reader_factory


def save_sparse_pack(
    path: Path,
    indices: np.ndarray,
    values: np.ndarray,
    nnz_ptr: np.ndarray,
    size: tuple[int, int],
    *,
    format: SparseFormat = SparseFormat.COO,
    dtype: np.dtype | None = None,
    value_scale: float = 1.0,
    manifest: PackManifest | None = None,
    files: PackFiles | None = None,
) -> None:
    """Write a sparse pack to disk using the registered codec for ``format``.

    Args:
        path: Destination directory.
        indices: COO indices array of shape (2, total_nnz).
        values: COO values array of shape (total_nnz,).
        nnz_ptr: Row pointer array of shape (n_samples + 1,).
        size: Matrix dimensions (rows, cols).
        format: Storage format; selects the codec via the registry.
        dtype: Force a specific numpy dtype for stored values.
        value_scale: Normalization scale; must be finite and > 0.
        manifest: Authoritative contract; overrides ``size``/``dtype``/``value_scale``.
        files: Custom payload filenames; ignored if ``manifest`` is provided.

    Raises:
        ValueError: If the format is not registered or validation fails.
    """
    get_codec(format).save(
        path,
        indices,
        values,
        nnz_ptr,
        size,
        dtype=dtype,
        value_scale=value_scale,
        manifest=manifest,
        files=files,
    )


def open_sparse_pack(
    path: Path,
    *,
    format: SparseFormat = SparseFormat.COO,
    manifest: PackManifest | None = None,
    files: PackFiles | None = None,
    matrix_size: tuple[int, int] | None = None,
    dtype: np.dtype | str | None = None,
) -> AbstractSparsePackReader:
    """Open a sparse pack and return a stateful reader.

    Payload arrays are loaded once at construction; subsequent calls to
    ``build_torch_sparse`` and ``build_torch_sparse_batch`` are I/O-free.

    Args:
        path: Pack directory.
        format: Storage format; selects the reader via the registry.
        manifest: Authoritative contract; validated against the payload when given.
        files: Custom payload filenames; used only when ``manifest`` is None.
        matrix_size: Explicit matrix dimensions for manifest inference.
        dtype: Explicit dtype for manifest inference.

    Returns:
        An ``AbstractSparsePackReader`` (LSP-compliant) for the given format.

    Raises:
        ValueError: If the format is not registered.
    """
    return get_reader_factory(format)(
        path,
        manifest,
        files=files,
        matrix_size=matrix_size,
        dtype=dtype,
    )


def is_sparse_pack_dir(path: Path, *, files: PackFiles | None = None) -> bool:
    """Check whether a directory contains the required sparse payload files.

    Args:
        path: Directory to inspect.
        files: Custom payload filenames; defaults to ``PackFiles()``.

    Returns:
        True if all required payload files (indices, values, nnz_ptr) exist.
    """
    pack_dir = Path(path)
    if not pack_dir.is_dir():
        return False
    payload_files = files or PackFiles()
    return all(
        (pack_dir / filename).exists()
        for filename in (payload_files.indices, payload_files.values, payload_files.nnz_ptr)
    )
