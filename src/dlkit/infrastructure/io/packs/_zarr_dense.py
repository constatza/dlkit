"""Zarr v3 dense array pack writer and reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import zarr
from torch import Tensor
from zarr import Array as ZarrArray
from zarr.codecs import BloscCodec, BloscShuffle

from ._manifest import ArrayPackManifest
from ._protocols import AbstractArrayPackReader

ZARR_GROUP_NAME = "pack.zarr"
ZARR_MARKER = "zarr.json"  # zarr format-3 group sentinel
_DATA_ARRAY_NAME = "data"


def _densify(data: Any) -> np.ndarray:
    """Convert scipy sparse or pass-through numpy arrays.

    Args:
        data: A numpy array or scipy sparse matrix.

    Returns:
        Dense numpy array.
    """
    try:
        from scipy.sparse import issparse

        if issparse(data):
            return data.toarray()
    except ImportError:
        pass
    return np.asarray(data)


class ZarrDensePackWriter:
    """Streaming context-manager writer for dense array packs backed by zarr v3.

    Appends matrices one at a time (or in batches) to resizable zarr chunked
    arrays with no pre-allocation.  On ``close()`` a manifest is written into
    the zarr group attributes so that ``ZarrDensePackReader`` can recover all
    metadata without scanning the array.

    On-disk layout::

        path/
          pack.zarr/
            zarr.json         zarr format-3 group sentinel
            data/             zarr array (N, rows, cols), zstd+bitshuffle

    Args:
        path: Destination directory for the pack.
        size: Matrix dimensions ``(rows, cols)``.
        dtype: Value dtype for the stored array. Default ``np.float32``.
        chunk_size: Number of samples per zarr chunk. Default ``64``.

    Raises:
        RuntimeError: If ``write_sample`` or ``write_samples`` is called after
            ``close()``.

    Example::

        with ZarrDensePackWriter(path, size=(R, C)) as w:
            for matrix in source:
                w.write_sample(matrix)
    """

    def __init__(
        self,
        path: Path,
        size: tuple[int, int],
        *,
        dtype: np.dtype | type = np.float32,
        chunk_size: int = 64,
    ) -> None:
        self._path = Path(path)
        rows, cols = size
        self._rows = rows
        self._cols = cols
        self._dtype = np.dtype(dtype)
        self._chunk_size = chunk_size
        self._n_written: int = 0
        self._closed: bool = False

        self._path.mkdir(parents=True, exist_ok=True)
        zarr_path = str(self._path / ZARR_GROUP_NAME)
        self._group = zarr.open_group(zarr_path, mode="w")
        self._data_arr = self._group.create_array(
            _DATA_ARRAY_NAME,
            shape=(0, rows, cols),
            chunks=(chunk_size, rows, cols),
            dtype=self._dtype,
            compressors=[
                BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle),
            ],
        )

    def write_sample(self, data: np.ndarray) -> None:
        """Append one ``(rows, cols)`` matrix to the pack.

        Accepts scipy sparse matrices when scipy is installed (soft import).

        Args:
            data: Dense array of shape ``(rows, cols)``, or scipy sparse
                matrix of equivalent dimensions.

        Raises:
            RuntimeError: If called after ``close()``.
            ValueError: If the array shape does not match the declared
                ``(rows, cols)``.
        """
        if self._closed:
            raise RuntimeError("write_sample called on a closed ZarrDensePackWriter")
        dense = _densify(data)
        arr = np.asarray(dense, dtype=self._dtype)
        if arr.shape != (self._rows, self._cols):
            raise ValueError(f"Expected shape ({self._rows}, {self._cols}), got {arr.shape}")
        self._data_arr.resize((self._n_written + 1, self._rows, self._cols))
        self._data_arr[-1:] = arr[np.newaxis]
        self._n_written += 1

    def write_samples(self, data: np.ndarray) -> None:
        """Append a batch of ``(K, rows, cols)`` matrices to the pack.

        Accepts scipy sparse matrices in batched form when scipy is installed.

        Args:
            data: Dense array of shape ``(K, rows, cols)``, or list/array of
                scipy sparse matrices (stacked).

        Raises:
            RuntimeError: If called after ``close()``.
            ValueError: If the array shape is not ``(K, rows, cols)``.
        """
        if self._closed:
            raise RuntimeError("write_samples called on a closed ZarrDensePackWriter")
        dense = _densify(data)
        arr = np.asarray(dense, dtype=self._dtype)
        if arr.ndim != 3 or arr.shape[1:] != (self._rows, self._cols):
            raise ValueError(f"Expected shape (K, {self._rows}, {self._cols}), got {arr.shape}")
        k = arr.shape[0]
        new_n = self._n_written + k
        self._data_arr.resize((new_n, self._rows, self._cols))
        self._data_arr[-k:] = arr
        self._n_written = new_n

    def close(self) -> None:
        """Finalise: write manifest into zarr group attributes. Idempotent.

        Safe to call multiple times — subsequent calls are no-ops.

        Raises:
            RuntimeError: If closed without any samples having been written.
        """
        if self._closed:
            return
        if self._n_written == 0:
            raise RuntimeError("ZarrDensePackWriter closed with no samples written")
        self._closed = True
        manifest = ArrayPackManifest(
            n_samples=self._n_written,
            matrix_size=(self._rows, self._cols),
            dtype=self._dtype.name,
            chunk_size=self._chunk_size,
        )
        self._group.attrs["dlkit_manifest"] = manifest.model_dump(by_alias=True)

    def __enter__(self) -> ZarrDensePackWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class ZarrDensePackReader(AbstractArrayPackReader):
    """Immutable reader for zarr v3 dense array packs.

    Opens the zarr group in read-only mode (``mode='r'``) — writes through
    this instance are structurally impossible.  ``__slots__`` prevents
    attribute injection after construction.

    Prefer constructing via the module-level ``open_array_pack`` factory.

    Attributes:
        _group: Read-only zarr group handle.
        _data: Zarr array ``(N, rows, cols)``.
        _manifest: Validated ``ArrayPackManifest`` from group attributes.
    """

    __slots__ = ("_group", "_data", "_manifest")

    def __init__(self, path: Path) -> None:
        zarr_path = str(Path(path) / ZARR_GROUP_NAME)
        self._group: zarr.Group = zarr.open_group(zarr_path, mode="r")
        raw_attrs = self._group.attrs.get("dlkit_manifest")
        raw_manifest: dict[str, object] = cast(
            "dict[str, object]", raw_attrs if isinstance(raw_attrs, dict) else {}
        )
        self._manifest = ArrayPackManifest.from_attrs(raw_manifest)
        self._data: ZarrArray = cast(ZarrArray, self._group[_DATA_ARRAY_NAME])

    # ------------------------------------------------------------------
    # AbstractArrayPackReader interface
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of matrices stored in the pack."""
        return self._manifest.n_samples

    @property
    def matrix_size(self) -> tuple[int, int]:
        """Shared ``(rows, cols)`` shape."""
        return self._manifest.matrix_size

    def __getitem__(self, idx: int | list[int] | slice) -> Tensor:
        """Return dense tensor(s) for the given index.

        Single-sample broadcast: when the pack contains exactly one sample,
        any requested index (including list/slice) is silently resolved to
        index ``0`` so the single matrix is replicated across the batch
        dimension.

        Args:
            idx: ``int`` → ``Tensor[rows, cols]``;
                 ``list[int]`` or ``slice`` → ``Tensor[B, rows, cols]``.

        Returns:
            Dense float ``Tensor``.

        Raises:
            IndexError: If a scalar index is out of range for a multi-sample pack.
        """
        match idx:
            case int():
                return self._get_single(idx)
            case list():
                return self._get_list(idx)
            case slice():
                indices = list(range(*idx.indices(self.n_samples)))
                return self._get_list(indices)
            case _:
                raise TypeError(f"Unsupported index type: {type(idx)}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_index(self, i: int) -> int:
        """Resolve a sample index, applying broadcast for single-sample packs.

        Args:
            i: Requested sample index.

        Returns:
            Resolved index safe to pass to zarr.

        Raises:
            IndexError: If ``i`` is out of range for a multi-sample pack.
        """
        if self.n_samples == 1:
            return 0
        if i < 0:
            i = self.n_samples + i
        if i < 0 or i >= self.n_samples:
            raise IndexError(f"index {i} out of range for pack with {self.n_samples} samples")
        return i

    def _get_single(self, i: int) -> Tensor:
        """Retrieve one matrix as a 2-D dense tensor.

        Args:
            i: Sample index.

        Returns:
            Dense ``Tensor`` of shape ``(rows, cols)``.
        """
        resolved = self._resolve_index(i)
        arr = np.asarray(self._data[resolved])
        return torch.from_numpy(arr)

    def _get_list(self, indices: list[int]) -> Tensor:
        """Retrieve multiple matrices as a 3-D dense tensor.

        Args:
            indices: List of sample indices.

        Returns:
            Dense ``Tensor`` of shape ``(B, rows, cols)``.
        """
        resolved = [self._resolve_index(i) for i in indices]
        arr = np.asarray(self._data.oindex[np.asarray(resolved, dtype=np.intp)])
        return torch.from_numpy(arr)
