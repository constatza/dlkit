"""Tests for Hdf5LazyReader — lazy indexed reader for HDF5 datasets."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dlkit.common.sources import ArraySource
from dlkit.infrastructure.hdf5 import Hdf5LazyReader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hdf5_array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create an HDF5 file with a flat dataset and return its path and source data.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Tuple of (file path, numpy array of shape (20, 4)).
    """
    data = np.arange(80, dtype="float32").reshape(20, 4)
    p = tmp_path / "test.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("x", data=data)
    return p, data


@pytest.fixture
def hdf5_grouped(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create an HDF5 file with a nested group dataset.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Tuple of (file path, numpy array of shape (10, 3)).
    """
    data = np.arange(30, dtype="float32").reshape(10, 3)
    p = tmp_path / "grouped.h5"
    with h5py.File(p, "w") as f:
        grp = f.create_group("data/train")
        grp.create_dataset("features", data=data)
    return p, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_n_samples(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader.n_samples equals 20.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, _ = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    assert reader.n_samples == 20


def test_sample_shape(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader.sample_shape equals (4,).

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, _ = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    assert reader.sample_shape == (4,)


def test_getitem_int_returns_correct_sample(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader[0] returns a tensor that matches data[0].

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    assert torch.allclose(reader[0], torch.from_numpy(data[0]))


def test_getitem_list_sorted_returns_correct_values(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader[[1, 3, 7]] returns correct tensors in caller order.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    indices = [1, 3, 7]
    result = reader[indices]
    assert result.shape == (3, 4)
    assert torch.allclose(result, torch.from_numpy(data[indices]))


def test_getitem_list_unsorted_preserves_caller_order(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader with unsorted indices returns data in caller order (not sorted order).

    h5py requires monotonically increasing indices internally; the reader must
    sort, read, then inverse-sort to restore the caller's requested order.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    indices = [7, 1, 15, 3]
    result = reader[indices]
    expected = torch.from_numpy(data[indices])
    assert result.shape == (4, 4)
    assert torch.allclose(result, expected)


def test_getitem_slice_returns_batch(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """reader[2:5] returns a tensor of shape (3, 4).

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, _ = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    result = reader[2:5]
    assert result.shape == (3, 4)


def test_lazy_open_file_not_opened_before_first_access(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """_file is None before any data access, ensuring per-worker lazy open.

    Shape metadata is cached at init without holding an open handle, so
    _file must remain None until the first __getitem__ call.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, _ = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    # n_samples / sample_shape are now cached — _file must still be None
    _ = reader.n_samples
    _ = reader.sample_shape
    assert reader._file is None
    _ = reader[0]
    assert reader._file is not None


def test_pickle_round_trip_preserves_data(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """Pickling a reader (e.g. for DataLoader spawn workers) round-trips correctly.

    _file must be None before pickling so h5py objects are never serialised.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    import pickle

    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    # Trigger n_samples access (as _maybe_broadcast does during dataset build)
    assert reader.n_samples == 20
    assert reader._file is None  # must not have opened the file
    unpickled = pickle.loads(pickle.dumps(reader))
    assert unpickled._file is None
    assert torch.allclose(unpickled[0], torch.from_numpy(data[0]))


def test_satisfies_array_source_protocol(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """Hdf5LazyReader is recognised as an ArraySource at runtime.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, _ = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    assert isinstance(reader, ArraySource)


def test_get_item_returns_single_sample(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """get_item(idx) returns a 1-D tensor matching the source sample.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    result = reader.get_item(5)
    assert result.shape == torch.Size([4])
    assert torch.allclose(result.float(), torch.from_numpy(data[5]).float())


def test_get_batch_returns_batched_tensor(hdf5_array: tuple[Path, np.ndarray]) -> None:
    """get_batch(indices) returns a (B, *sample_shape) tensor.

    Args:
        hdf5_array: Fixture providing (path, data).
    """
    path, data = hdf5_array
    reader = Hdf5LazyReader(path, "x")
    indices = [0, 5, 19]
    result = reader.get_batch(indices)
    assert result.shape == torch.Size([3, 4])
    assert torch.allclose(result.float(), torch.from_numpy(data[indices]).float())


def test_nested_group_dataset_path(hdf5_grouped: tuple[Path, np.ndarray]) -> None:
    """Reader navigates nested groups via forward-slash dataset_path.

    Args:
        hdf5_grouped: Fixture providing (path, data) for a nested-group HDF5 file.
    """
    path, data = hdf5_grouped
    reader = Hdf5LazyReader(path, "data/train/features")
    assert reader.n_samples == 10
    assert reader.sample_shape == (3,)
    assert torch.allclose(reader[0], torch.from_numpy(data[0]))
