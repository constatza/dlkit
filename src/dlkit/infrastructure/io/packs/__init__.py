"""Dense array pack I/O backed by zarr v3.

Replaces the old COO sparse format with compressed dense arrays.  Every sample
is a ``(rows, cols)`` matrix; the pack stores ``N`` such matrices in a single
zarr group with zstd+bitshuffle compression.

Two API tiers are provided:

**Open (lazy reader handle):**

    open_array_pack(path)           ->  AbstractArrayPackReader  # auto-detect
    ZarrDensePackReader(path)       ->  ZarrDensePackReader      # explicit

**Streaming writer factories (return a context-manager writer):**

    write_array_pack(path, size)    ->  IArrayPackWriter  # registry-dispatched
    ZarrDensePackWriter(path, size) ->  ZarrDensePackWriter

**Batch write:**

    save_array_pack(path, data)     ->  None  (data: np.ndarray shape (N,R,C))

Format auto-detection:
    ``open_array_pack`` detects zarr-dense by checking for
    ``pack.zarr/zarr.json``; raises ``ValueError`` for unrecognised layouts.

Registry extension (OCP):
    Third-party formats can be registered without modifying library code::

        register_format(ArrayPackFormat.MY_FORMAT, reader_factory, writer_factory=my_writer_factory)

Downstream usage::

    reader = open_array_pack(path)
    t = reader[0]  # -> Tensor (rows, cols)
    batch = reader[[0, 2]]  # -> Tensor (2, rows, cols)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._factory import detect_format, open_array_pack, save_array_pack, write_array_pack
from ._manifest import ARRAY_PACK_SCHEMA, ArrayPackManifest
from ._protocols import (
    AbstractArrayPackReader,
    ArrayPackFormat,
    IArrayPackReader,
    IArrayPackWriter,
)
from ._registry import register_format
from ._zarr_dense import ZarrDensePackReader, ZarrDensePackWriter


def _open_zarr_dense_pack(path: Path) -> ZarrDensePackReader:
    """Registry adapter: open a ``ZarrDensePackReader`` from a directory.

    Args:
        path: Pack directory containing ``pack.zarr/``.

    Returns:
        Fully initialised ``ZarrDensePackReader``.
    """
    return ZarrDensePackReader(Path(path))


def _open_zarr_dense_pack_writer(
    path: Path,
    size: tuple[int, int],
    *,
    dtype: np.dtype | type = np.float32,
    chunk_size: int = 64,
) -> ZarrDensePackWriter:
    """Registry adapter: construct a ``ZarrDensePackWriter``.

    Args:
        path: Destination directory.
        size: Matrix dimensions ``(rows, cols)``.
        dtype: Value dtype for the stored array.
        chunk_size: Number of samples per zarr chunk.

    Returns:
        ``ZarrDensePackWriter`` context manager.
    """
    return ZarrDensePackWriter(Path(path), size, dtype=dtype, chunk_size=chunk_size)


# Register the built-in zarr-dense format at import time.
# Adding a new format requires only creating a new module and calling
# register_format here — zero changes to _factory.py or _protocols.py.
register_format(
    ArrayPackFormat.ZARR_DENSE,
    _open_zarr_dense_pack,
    writer_factory=_open_zarr_dense_pack_writer,
)

__all__ = [
    "ARRAY_PACK_SCHEMA",
    "AbstractArrayPackReader",
    "ArrayPackFormat",
    "ArrayPackManifest",
    "IArrayPackReader",
    "IArrayPackWriter",
    "ZarrDensePackReader",
    "ZarrDensePackWriter",
    "detect_format",
    "open_array_pack",
    "register_format",
    "save_array_pack",
    "write_array_pack",
]
