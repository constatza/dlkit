"""Format-agnostic dense array pack factory functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._protocols import AbstractArrayPackReader, ArrayPackFormat, IArrayPackWriter
from ._registry import get_reader_factory, get_writer_factory
from ._zarr_dense import ZARR_GROUP_NAME as _ZARR_GROUP_NAME
from ._zarr_dense import ZARR_MARKER as _ZARR_MARKER


def detect_format(path: Path) -> ArrayPackFormat:
    """Detect the storage format of an existing array pack directory.

    Currently the only supported detection heuristic checks for the zarr v3
    group sentinel ``pack.zarr/zarr.json``.  Raise ``ValueError`` when no
    known format is detected.

    Args:
        path: Directory to inspect.

    Returns:
        Detected ``ArrayPackFormat``.

    Raises:
        ValueError: If no recognised pack format is found at ``path``.
    """
    if (Path(path) / _ZARR_GROUP_NAME / _ZARR_MARKER).exists():
        return ArrayPackFormat.ZARR_DENSE
    raise ValueError(
        f"Cannot detect array pack format at '{path}'. "
        f"Expected '{_ZARR_GROUP_NAME}/{_ZARR_MARKER}' for zarr-dense format."
    )


def open_array_pack(
    path: Path,
    *,
    format: ArrayPackFormat | None = None,
) -> AbstractArrayPackReader:
    """Open a dense array pack and return a stateful reader.

    Auto-detects the storage format when ``format`` is ``None`` by inspecting
    the directory layout.  Pass an explicit ``format`` to override detection.

    Args:
        path: Pack directory.
        format: Storage format override.  When ``None`` the format is
            auto-detected via ``detect_format``.

    Returns:
        An ``AbstractArrayPackReader`` (LSP-compliant) for the given format.

    Raises:
        ValueError: If the format is not registered or cannot be detected.
    """
    resolved_format = format if format is not None else detect_format(path)
    return get_reader_factory(resolved_format)(path)


def write_array_pack(
    path: Path,
    size: tuple[int, int],
    *,
    format: ArrayPackFormat = ArrayPackFormat.ZARR_DENSE,
    dtype: np.dtype | type = np.float32,
    chunk_size: int = 64,
) -> IArrayPackWriter:
    """Return a streaming context-manager writer for a dense array pack.

    Use as a context manager.  On ``__exit__`` the pack is finalised and the
    manifest is written so the pack is immediately readable via
    ``open_array_pack``.

    Args:
        path: Destination directory.
        size: Matrix dimensions ``(rows, cols)``.
        format: Storage format; selects the writer via the registry.  Default
            is ``ArrayPackFormat.ZARR_DENSE``.
        dtype: Value dtype for the stored array.  Default ``np.float32``.
        chunk_size: Number of samples per zarr chunk.  Default ``64``.

    Returns:
        An ``IArrayPackWriter`` context manager.

    Raises:
        ValueError: If no writer factory is registered for ``format``.

    Example::

        with write_array_pack(path, size=(R, C)) as w:
            for matrix in source:
                w.write_sample(matrix)
    """
    return get_writer_factory(format)(path, size, dtype=dtype, chunk_size=chunk_size)


def save_array_pack(
    path: Path,
    data: np.ndarray,
    *,
    format: ArrayPackFormat = ArrayPackFormat.ZARR_DENSE,
    dtype: np.dtype | type | None = None,
    chunk_size: int = 64,
) -> None:
    """Batch-write an array pack from a pre-assembled ``(N, rows, cols)`` array.

    Internally opens a ``write_array_pack`` context manager and calls
    ``write_samples`` once.  Prefer ``write_array_pack`` for streaming use
    cases where the full array never resides in RAM simultaneously.

    Args:
        path: Destination directory.
        data: Dense numpy array of shape ``(N, rows, cols)``.
        format: Storage format.  Default ``ArrayPackFormat.ZARR_DENSE``.
        dtype: Value dtype override.  When ``None`` the dtype is inferred from
            ``data.dtype``.
        chunk_size: Number of samples per zarr chunk.  Default ``64``.

    Raises:
        ValueError: If ``data`` is not 3-D.
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3-D (N, rows, cols), got shape {data.shape}")
    resolved_dtype: np.dtype | type = dtype if dtype is not None else data.dtype
    _, rows, cols = data.shape
    with write_array_pack(
        path,
        size=(rows, cols),
        format=format,
        dtype=resolved_dtype,
        chunk_size=chunk_size,
    ) as writer:
        writer.write_samples(data)
