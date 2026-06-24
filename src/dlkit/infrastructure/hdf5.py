"""HDF5 IO abstractions — ``Hdf5LazyReader`` for lazy per-sample indexed access.

Lives at the ``infrastructure`` level (parallel to ``infrastructure/zarr.py``)
so that ``infrastructure.config`` entry types can return ``Hdf5LazyReader``
instances without creating a ``config→io`` import cycle.

Shape metadata is probed once at construction (brief open + close).  The data
file itself is only opened on first ``__getitem__`` call, so ``_file`` is
always ``None`` at pickle time and each DataLoader worker opens its own
independent file descriptor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class Hdf5LazyReader:
    """Lazy indexed reader for a single HDF5 dataset.

    Shape metadata is probed once at construction (brief open + close) so that
    ``n_samples`` and ``sample_shape`` never require an open file handle.  The
    data file itself is only opened on first ``__getitem__`` call, ensuring
    ``_file`` is always ``None`` at pickle time and each DataLoader worker opens
    its own independent file descriptor.

    Args:
        path: Path to the HDF5 file.
        dataset_path: Full dataset path within the file, e.g. ``"arrays/x"``.
    """

    def __init__(self, path: Path, dataset_path: str) -> None:
        import h5py

        self._path = path
        self._dataset_path = dataset_path
        self._file: Any | None = None
        with h5py.File(path, "r") as f:
            ds = f[dataset_path]
            self._n_samples: int = ds.shape[0]
            self._sample_shape: tuple[int, ...] = tuple(ds.shape[1:])

    def _ds(self) -> Any:
        """Return the open HDF5 dataset, opening the file on first access.

        Returns:
            h5py Dataset object for the configured dataset path.
        """
        if self._file is None:
            import h5py

            self._file = h5py.File(self._path, "r")
        return self._file[self._dataset_path]

    def __del__(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def __getitem__(self, idx: int | list[int] | slice) -> torch.Tensor:
        """Read sample(s) by index.

        For ``list[int]``: h5py fancy indexing requires monotonically increasing
        indices (HDF5 C library constraint). Sort → read → inverse-sort
        restores the caller's original order without extra copies.

        Args:
            idx: Single index, list of indices, or slice.

        Returns:
            Tensor of shape ``(*sample_shape,)`` for int, or
            ``(n, *sample_shape)`` for list/slice.
        """
        ds = self._ds()
        match idx:
            case int():
                return torch.from_numpy(np.array(ds[idx]))
            case []:
                return torch.from_numpy(np.array(ds[0:0]))
            case list():
                arr = np.asarray(idx)
                order = np.argsort(arr)
                inv = np.argsort(order)
                data = np.array(ds[arr[order].tolist()])[inv]
                return torch.from_numpy(data.copy())
            case slice():
                return torch.from_numpy(np.array(ds[idx]))

    @property
    def n_samples(self) -> int:
        """Total number of samples (length of axis 0).

        Returns:
            Integer sample count.
        """
        return self._n_samples

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of a single sample (all dimensions after axis 0).

        Returns:
            Tuple of ints.
        """
        return self._sample_shape

    def get_item(self, idx: int) -> torch.Tensor:
        """Return a single sample cast to the configured precision dtype.

        Args:
            idx: Zero-based sample index.

        Returns:
            Tensor of shape ``(*sample_shape,)``.
        """
        from dlkit.infrastructure.precision.service import PrecisionService

        return PrecisionService().cast_tensor(self[idx])

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        """Return a batch tensor for the given indices cast to precision dtype.

        Args:
            indices: List of zero-based sample indices.

        Returns:
            Tensor of shape ``(B, *sample_shape)`` where ``B = len(indices)``.
        """
        from dlkit.infrastructure.precision.service import PrecisionService

        return PrecisionService().cast_tensor(self[indices])
