"""Tests for PathFeature zarr directory validation (formerly MatrixFeature/SparseFeature)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.infrastructure.config.entry_factories import Feature
from dlkit.infrastructure.config.entry_types import PathFeature
from dlkit.infrastructure.io.packs import save_array_pack

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
    """Create a valid zarr array pack directory.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the zarr pack root directory.
    """
    rng = np.random.default_rng(0)
    data = rng.random((3, 4, 4)).astype(np.float32)
    pack_path = tmp_path / "valid_pack.zarr"
    save_array_pack(pack_path, data)
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
# PathFeature — zarr directory validation
# ---------------------------------------------------------------------------


def test_path_feature_accepts_zarr_dir(zarr_pack_dir: Path) -> None:
    """Feature() pointing at a valid zarr pack directory validates successfully.

    Args:
        zarr_pack_dir: Valid zarr array pack directory.
    """
    feature = Feature(name="K", path=zarr_pack_dir)
    assert isinstance(feature, PathFeature)
    assert feature.path == zarr_pack_dir
    assert feature.name == "K"


def test_path_feature_direct_constructs_with_zarr_dir(zarr_pack_dir: Path) -> None:
    """PathFeature constructed directly also accepts a valid zarr directory.

    Args:
        zarr_pack_dir: Valid zarr array pack directory.
    """
    feature = PathFeature(name="K", path=zarr_pack_dir)
    assert feature.path == zarr_pack_dir
    assert feature.name == "K"


def test_path_feature_rejects_non_zarr_dir(plain_dir: Path) -> None:
    """Feature() pointing at a plain directory (not a zarr pack) raises ValueError.

    Args:
        plain_dir: A regular directory with no zarr metadata.
    """
    with pytest.raises(ValueError):
        Feature(name="K", path=plain_dir)


def test_path_feature_accepts_file(npy_file: Path) -> None:
    """Feature() pointing at a plain .npy file validates successfully.

    Args:
        npy_file: Path to an existing .npy file.
    """
    feature = Feature(name="x", path=npy_file)
    assert isinstance(feature, PathFeature)
    assert feature.path == npy_file


def test_path_feature_accepts_none_path() -> None:
    """PathFeature with no path is a valid placeholder."""
    feature = PathFeature(name="K")
    assert feature.path is None
    assert feature.is_placeholder()


def test_path_feature_accepts_non_directory_file(tmp_path: Path) -> None:
    """PathFeature accepts any existing file path regardless of extension.

    The zarr-pack validator only fires when the path is a directory;
    a plain file is treated as a regular array file.

    Args:
        tmp_path: pytest temporary directory.
    """
    file_path = tmp_path / "data.npy"
    np.save(file_path, np.ones((5, 3), dtype=np.float32))

    feature = PathFeature(name="K", path=file_path)
    assert feature.path == file_path


def test_path_feature_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    """PathFeature rejects a path that does not exist on disk.

    Args:
        tmp_path: pytest temporary directory.
    """
    missing = tmp_path / "nonexistent.zarr"

    with pytest.raises(ValueError, match="does not exist"):
        PathFeature(name="K", path=missing)


def test_path_feature_has_no_files_field() -> None:
    """PathFeature must not expose a legacy ``files`` field."""
    assert "files" not in PathFeature.model_fields
