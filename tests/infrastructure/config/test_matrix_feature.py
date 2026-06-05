"""Tests for ZarrEntry zarr directory validation (formerly MatrixFeature/SparseFeature)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry, PathBasedEntry, ZarrEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def plain_dir(tmp_path: Path) -> Path:
    """Create a plain directory that is NOT a zarr pack.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the plain (non-zarr) directory.
    """
    d = tmp_path / "not_a_zarr"
    d.mkdir()
    return d


@pytest.fixture
def zarr_pack_dir(tmp_path: Path) -> Path:
    """Create a valid native zarr array store directory.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the native zarr store directory.
    """
    rng = np.random.default_rng(0)
    data = rng.random((3, 4, 4)).astype(np.float32)
    pack_path = tmp_path / "valid_pack.zarr"
    arr = zarr.open_array(
        str(pack_path), mode="w", shape=data.shape, chunks=(1, 4, 4), dtype=data.dtype
    )
    arr[:] = data
    return pack_path


@pytest.fixture
def npy_file(tmp_path: Path) -> Path:
    """Create a simple .npy file.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the .npy file.
    """
    path = tmp_path / "data.npy"
    np.save(path, np.ones((10, 5), dtype=np.float32))
    return path


# ---------------------------------------------------------------------------
# ZarrEntry — zarr directory validation
# ---------------------------------------------------------------------------


def test_zarr_entry_accepts_zarr_dir(zarr_pack_dir: Path) -> None:
    """ZarrEntry pointing at a valid zarr pack directory validates successfully.

    Args:
        zarr_pack_dir: Valid zarr array pack directory.
    """
    feature = ZarrEntry(name="K", path=zarr_pack_dir, data_role=DataRole.FEATURE)
    assert isinstance(feature, PathBasedEntry)
    assert feature.path == zarr_pack_dir
    assert feature.name == "K"
    assert feature.data_role == DataRole.FEATURE


def test_zarr_entry_direct_constructs_with_zarr_dir(zarr_pack_dir: Path) -> None:
    """ZarrEntry constructed directly also accepts a valid zarr directory.

    Args:
        zarr_pack_dir: Valid zarr array pack directory.
    """
    feature = ZarrEntry(name="K", path=zarr_pack_dir, data_role=DataRole.FEATURE)
    assert feature.path == zarr_pack_dir
    assert feature.name == "K"


def test_zarr_entry_rejects_non_zarr_dir(plain_dir: Path) -> None:
    """ZarrEntry pointing at a plain directory (not a zarr pack) raises ValueError.

    Args:
        plain_dir: A regular directory with no zarr metadata.
    """
    with pytest.raises(ValueError):
        ZarrEntry(name="K", path=plain_dir, data_role=DataRole.FEATURE)


def test_npy_entry_accepts_file(npy_file: Path) -> None:
    """NpyEntry pointing at a plain .npy file validates successfully.

    Args:
        npy_file: Path to an existing .npy file.
    """
    feature = NpyEntry(name="x", path=npy_file, data_role=DataRole.FEATURE)
    assert isinstance(feature, PathBasedEntry)
    assert feature.path == npy_file


def test_zarr_entry_accepts_none_path() -> None:
    """ZarrEntry with no path is a valid placeholder."""
    feature = ZarrEntry(name="K", data_role=DataRole.FEATURE)
    assert feature.path is None
    assert feature.is_placeholder()


def test_npy_entry_accepts_non_directory_file(npy_file: Path) -> None:
    """NpyEntry accepts any existing .npy file path.

    Args:
        npy_file: Path to an existing .npy file.
    """
    feature = NpyEntry(name="K", path=npy_file, data_role=DataRole.FEATURE)
    assert feature.path == npy_file


def test_zarr_entry_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """ZarrEntry rejects a path that does not exist on disk.

    Args:
        tmp_path: pytest temporary directory.
    """
    missing = tmp_path / "nonexistent.zarr"

    with pytest.raises(ValueError, match="does not exist"):
        ZarrEntry(name="K", path=missing, data_role=DataRole.FEATURE)


def test_zarr_entry_has_no_files_field() -> None:
    """ZarrEntry must not expose a legacy ``files`` field."""
    assert "files" not in ZarrEntry.model_fields
