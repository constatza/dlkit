"""Tests for format-specific DataEntry subclasses.

Covers suffix validation, load_kwargs, defaults, and zarr store detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import (
    CsvEntry,
    Hdf5Entry,
    NpyEntry,
    NpzEntry,
    ParquetEntry,
    ValueEntry,
    ZarrEntry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def npy_path(tmp_path: Path) -> Path:
    """Create a valid .npy file.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the written .npy file.
    """
    p = tmp_path / "data.npy"
    np.save(str(p), np.random.randn(10, 5))
    return p


@pytest.fixture
def npz_path(tmp_path: Path) -> Path:
    """Create a valid .npz file.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the written .npz file.
    """
    p = tmp_path / "data.npz"
    np.savez(str(p), x=np.random.randn(10, 5))
    return p


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    """Create a minimal CSV file.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the written .csv file.
    """
    p = tmp_path / "data.csv"
    p.write_text("a,b,c\n1,2,3\n")
    return p


@pytest.fixture
def zarr_store_path(tmp_path: Path) -> Path:
    """Create a valid native zarr array store.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the zarr array directory.
    """
    p = tmp_path / "array.zarr"
    z = zarr.open_array(str(p), mode="w", shape=(10, 5), chunks=(1, 5), dtype="float32")
    z[:] = np.random.randn(10, 5)
    return p


@pytest.fixture
def parquet_path(tmp_path: Path) -> Path:
    """Create a stub .parquet file (suffix validation only).

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the stub .parquet file.
    """
    p = tmp_path / "data.parquet"
    p.touch()
    return p


@pytest.fixture
def hdf5_path(tmp_path: Path) -> Path:
    """Create a minimal HDF5 file with an ``arrays/x`` dataset.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the HDF5 file.
    """
    import h5py
    import numpy as np

    p = tmp_path / "data.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("arrays/x", data=np.zeros((10, 4), dtype="float32"))
    return p


# ---------------------------------------------------------------------------
# NpyEntry tests
# ---------------------------------------------------------------------------


def test_npy_entry_accepts_valid_path(npy_path: Path) -> None:
    """NpyEntry instantiates without error for a .npy file.

    Args:
        npy_path: Path to a valid .npy file.
    """
    entry = NpyEntry(name="feat", path=npy_path)
    assert entry.path == npy_path


def test_npy_entry_rejects_wrong_suffix(tmp_path: Path) -> None:
    """NpyEntry raises ValueError when suffix is not .npy.

    Args:
        tmp_path: pytest temporary directory.
    """
    p = tmp_path / "data.csv"
    p.touch()
    with pytest.raises(ValueError, match=r"\.npy"):
        NpyEntry(name="feat", path=p)


def test_npy_entry_load_kwargs_mmap_true(npy_path: Path) -> None:
    """NpyEntry.load_kwargs returns mmap_mode='r' when mmap=True.

    Args:
        npy_path: Path to a valid .npy file.
    """
    entry = NpyEntry(name="feat", path=npy_path, mmap=True)
    assert entry.load_kwargs == {"mmap_mode": "r"}


def test_npy_entry_load_kwargs_mmap_false(npy_path: Path) -> None:
    """NpyEntry.load_kwargs returns empty dict when mmap=False.

    Args:
        npy_path: Path to a valid .npy file.
    """
    entry = NpyEntry(name="feat", path=npy_path, mmap=False)
    assert entry.load_kwargs == {}


def test_npy_entry_data_role_default(npy_path: Path) -> None:
    """NpyEntry.data_role defaults to DataRole.FEATURE.

    Args:
        npy_path: Path to a valid .npy file.
    """
    entry = NpyEntry(name="feat", path=npy_path)
    assert entry.data_role == DataRole.FEATURE


def test_npy_entry_write_default(npy_path: Path) -> None:
    """NpyEntry.write defaults to False.

    Args:
        npy_path: Path to a valid .npy file.
    """
    entry = NpyEntry(name="feat", path=npy_path)
    assert entry.write is False


# ---------------------------------------------------------------------------
# NpzEntry tests
# ---------------------------------------------------------------------------


def test_npz_entry_accepts_valid_path(npz_path: Path) -> None:
    """NpzEntry instantiates without error for a .npz file.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path)
    assert entry.path == npz_path


def test_npz_entry_rejects_wrong_suffix(tmp_path: Path) -> None:
    """NpzEntry raises ValueError when suffix is not .npz.

    Args:
        tmp_path: pytest temporary directory.
    """
    p = tmp_path / "data.npy"
    np.save(str(p), np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"\.npz"):
        NpzEntry(name="feat", path=p)


def test_npz_entry_load_kwargs_mmap_true(npz_path: Path) -> None:
    """NpzEntry.load_kwargs returns mmap_mode='r' when mmap=True.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path, mmap=True)
    assert entry.load_kwargs == {"mmap_mode": "r"}


def test_npz_entry_load_kwargs_mmap_false(npz_path: Path) -> None:
    """NpzEntry.load_kwargs returns empty dict when mmap=False.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path, mmap=False)
    assert entry.load_kwargs == {}


def test_npz_entry_key_defaults_to_none(npz_path: Path) -> None:
    """NpzEntry.key defaults to None.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path)
    assert entry.key is None


def test_npz_entry_data_role_default(npz_path: Path) -> None:
    """NpzEntry.data_role defaults to DataRole.FEATURE.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path)
    assert entry.data_role == DataRole.FEATURE


def test_npz_entry_write_default(npz_path: Path) -> None:
    """NpzEntry.write defaults to False.

    Args:
        npz_path: Path to a valid .npz file.
    """
    entry = NpzEntry(name="feat", path=npz_path)
    assert entry.write is False


# ---------------------------------------------------------------------------
# CsvEntry tests
# ---------------------------------------------------------------------------


def test_csv_entry_accepts_valid_path(csv_path: Path) -> None:
    """CsvEntry instantiates without error for a .csv file.

    Args:
        csv_path: Path to a valid .csv file.
    """
    entry = CsvEntry(name="feat", path=csv_path)
    assert entry.path == csv_path


def test_csv_entry_rejects_wrong_suffix(tmp_path: Path) -> None:
    """CsvEntry raises ValueError when suffix is not .csv or .txt.

    Args:
        tmp_path: pytest temporary directory.
    """
    p = tmp_path / "data.parquet"
    p.touch()
    with pytest.raises(ValueError, match=r"csv"):
        CsvEntry(name="feat", path=p)


def test_csv_entry_data_role_default(csv_path: Path) -> None:
    """CsvEntry.data_role defaults to DataRole.FEATURE.

    Args:
        csv_path: Path to a valid .csv file.
    """
    entry = CsvEntry(name="feat", path=csv_path)
    assert entry.data_role == DataRole.FEATURE


def test_csv_entry_write_default(csv_path: Path) -> None:
    """CsvEntry.write defaults to False.

    Args:
        csv_path: Path to a valid .csv file.
    """
    entry = CsvEntry(name="feat", path=csv_path)
    assert entry.write is False


# ---------------------------------------------------------------------------
# ZarrEntry tests
# ---------------------------------------------------------------------------


def test_zarr_entry_accepts_valid_store(zarr_store_path: Path) -> None:
    """ZarrEntry instantiates without error for a valid zarr store.

    Args:
        zarr_store_path: Path to a valid zarr array directory.
    """
    entry = ZarrEntry(name="feat", path=zarr_store_path)
    assert entry.path == zarr_store_path


def test_zarr_entry_validate_zarr_store_raises_for_missing_sentinel(
    tmp_path: Path,
) -> None:
    """ZarrEntry._validate_zarr_store raises ValueError for dirs without zarr.json.

    Args:
        tmp_path: pytest temporary directory.
    """
    bad_dir = tmp_path / "not_a_zarr"
    bad_dir.mkdir()
    with pytest.raises(ValueError, match=r"zarr\.json"):
        ZarrEntry(name="feat", path=bad_dir)


def test_zarr_entry_open_reader_returns_lazy_reader(zarr_store_path: Path) -> None:
    """ZarrEntry.open_reader() returns an ILazyReader instance.

    Args:
        zarr_store_path: Path to a valid zarr array directory.
    """
    from dlkit.infrastructure.zarr import ILazyReader

    entry = ZarrEntry(name="feat", path=zarr_store_path)
    result = entry.open_reader()
    assert isinstance(result, ILazyReader)


def test_zarr_entry_data_role_default(zarr_store_path: Path) -> None:
    """ZarrEntry.data_role defaults to DataRole.FEATURE.

    Args:
        zarr_store_path: Path to a valid zarr array directory.
    """
    entry = ZarrEntry(name="feat", path=zarr_store_path)
    assert entry.data_role == DataRole.FEATURE


def test_zarr_entry_write_default(zarr_store_path: Path) -> None:
    """ZarrEntry.write defaults to False.

    Args:
        zarr_store_path: Path to a valid zarr array directory.
    """
    entry = ZarrEntry(name="feat", path=zarr_store_path)
    assert entry.write is False


# ---------------------------------------------------------------------------
# ParquetEntry tests
# ---------------------------------------------------------------------------


def test_parquet_entry_accepts_valid_path(parquet_path: Path) -> None:
    """ParquetEntry instantiates without error for a .parquet file.

    Args:
        parquet_path: Path to a stub .parquet file.
    """
    entry = ParquetEntry(name="feat", path=parquet_path)
    assert entry.path == parquet_path


def test_parquet_entry_rejects_wrong_suffix(tmp_path: Path) -> None:
    """ParquetEntry raises ValueError when suffix is not .parquet.

    Args:
        tmp_path: pytest temporary directory.
    """
    p = tmp_path / "data.csv"
    p.touch()
    with pytest.raises(ValueError, match=r"parquet"):
        ParquetEntry(name="feat", path=p)


def test_parquet_entry_data_role_default(parquet_path: Path) -> None:
    """ParquetEntry.data_role defaults to DataRole.FEATURE.

    Args:
        parquet_path: Path to a stub .parquet file.
    """
    entry = ParquetEntry(name="feat", path=parquet_path)
    assert entry.data_role == DataRole.FEATURE


def test_parquet_entry_write_default(parquet_path: Path) -> None:
    """ParquetEntry.write defaults to False.

    Args:
        parquet_path: Path to a stub .parquet file.
    """
    entry = ParquetEntry(name="feat", path=parquet_path)
    assert entry.write is False


# ---------------------------------------------------------------------------
# Hdf5Entry tests
# ---------------------------------------------------------------------------


def test_hdf5_entry_accepts_valid_path(hdf5_path: Path) -> None:
    """Hdf5Entry instantiates without error for a .h5 file.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path)
    assert entry.path == hdf5_path


def test_hdf5_entry_rejects_wrong_suffix(tmp_path: Path) -> None:
    """Hdf5Entry raises ValueError when suffix is not .h5 or .hdf5.

    Args:
        tmp_path: pytest temporary directory.
    """
    p = tmp_path / "data.npy"
    np.save(str(p), np.ones((2, 2)))
    with pytest.raises(ValueError, match=r"hdf5|h5"):
        Hdf5Entry(name="feat", path=p)


def test_hdf5_entry_data_role_default(hdf5_path: Path) -> None:
    """Hdf5Entry.data_role defaults to DataRole.FEATURE.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path)
    assert entry.data_role == DataRole.FEATURE


def test_hdf5_entry_write_default(hdf5_path: Path) -> None:
    """Hdf5Entry.write defaults to False.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path)
    assert entry.write is False


def test_hdf5_entry_lazy_default(hdf5_path: Path) -> None:
    """Hdf5Entry.lazy defaults to True (lazy loading enabled).

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path)
    assert entry.lazy is True


def test_hdf5_entry_is_multi_array(hdf5_path: Path) -> None:
    """Hdf5Entry.is_multi_array is always True so dispatch passes array_key.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path)
    assert entry.is_multi_array is True


def test_hdf5_entry_array_key_no_group(hdf5_path: Path) -> None:
    """array_key equals key when group is None.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path, key="y")
    assert entry.array_key == "y"


def test_hdf5_entry_array_key_with_group(hdf5_path: Path) -> None:
    """array_key equals 'group/key' when group is set.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path, group="arrays", key="x")
    assert entry.array_key == "arrays/x"


def test_hdf5_entry_array_key_nested_group(hdf5_path: Path) -> None:
    """array_key handles nested group paths correctly.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path, group="data/train", key="features")
    assert entry.array_key == "data/train/features"


def test_hdf5_entry_open_reader_lazy_returns_lazy_reader(hdf5_path: Path) -> None:
    """open_reader() returns Hdf5LazyReader when lazy=True.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    from dlkit.infrastructure.hdf5 import Hdf5LazyReader

    entry = Hdf5Entry(name="feat", path=hdf5_path, group="arrays", key="x")
    reader = entry.open_reader()
    assert isinstance(reader, Hdf5LazyReader)


def test_hdf5_entry_open_reader_eager_returns_path(hdf5_path: Path) -> None:
    """open_reader() returns Path when lazy=False.

    Args:
        hdf5_path: Path to a stub .h5 file.
    """
    entry = Hdf5Entry(name="feat", path=hdf5_path, lazy=False)
    reader = entry.open_reader()
    assert reader == hdf5_path


# ---------------------------------------------------------------------------
# ValueEntry tests
# ---------------------------------------------------------------------------


def test_value_entry_data_role_default() -> None:
    """ValueEntry.data_role defaults to DataRole.FEATURE."""
    entry = ValueEntry(name="feat")
    assert entry.data_role == DataRole.FEATURE


def test_value_entry_write_default() -> None:
    """ValueEntry.write defaults to False."""
    entry = ValueEntry(name="feat")
    assert entry.write is False
