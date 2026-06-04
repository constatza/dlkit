"""Tests for MatrixFeature data entry and backward-compatible SparseFeature alias."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.infrastructure.config.entry_factories import Matrix, Sparse
from dlkit.infrastructure.config.entry_types import MatrixFeature, SparseFeature

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pack_dir(tmp_path: Path) -> Path:
    """Create a minimal directory to simulate a zarr pack root.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the created directory.
    """
    d = tmp_path / "pack.zarr"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# MatrixFeature construction
# ---------------------------------------------------------------------------


def test_matrix_feature_constructs_with_valid_directory(pack_dir: Path) -> None:
    feature = MatrixFeature(name="K", path=pack_dir)
    assert feature.path == pack_dir
    assert feature.name == "K"


def test_matrix_feature_constructs_with_none_path() -> None:
    feature = MatrixFeature(name="K")
    assert feature.path is None
    assert feature.is_placeholder()


def test_matrix_feature_has_no_files_field() -> None:
    assert "files" not in MatrixFeature.model_fields


def test_matrix_feature_raises_when_path_is_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.zarr"
    file_path.write_bytes(b"dummy")

    with pytest.raises(ValueError, match="must be a directory"):
        MatrixFeature(name="K", path=file_path)


def test_matrix_feature_raises_when_path_does_not_exist(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.zarr"

    with pytest.raises(ValueError, match="does not exist"):
        MatrixFeature(name="K", path=missing)


# ---------------------------------------------------------------------------
# Matrix() factory
# ---------------------------------------------------------------------------


def test_matrix_factory_creates_matrix_feature(pack_dir: Path) -> None:
    feature = Matrix(name="K", path=pack_dir)
    assert isinstance(feature, MatrixFeature)
    assert feature.path == pack_dir


def test_matrix_factory_with_none_path_creates_placeholder() -> None:
    feature = Matrix(name="K")
    assert isinstance(feature, MatrixFeature)
    assert feature.path is None


def test_matrix_factory_accepts_string_path(pack_dir: Path) -> None:
    feature = Matrix(name="K", path=str(pack_dir))
    assert isinstance(feature, MatrixFeature)
    assert feature.path == pack_dir


# ---------------------------------------------------------------------------
# Backward compatibility — SparseFeature alias and Sparse() factory
# ---------------------------------------------------------------------------


def test_sparse_feature_is_alias_for_matrix_feature() -> None:
    assert SparseFeature is MatrixFeature


def test_sparse_feature_constructs_correctly(pack_dir: Path) -> None:
    feature = SparseFeature(name="matrix", path=pack_dir)
    assert isinstance(feature, MatrixFeature)
    assert feature.path == pack_dir


def test_sparse_feature_alias_placeholder() -> None:
    feature = SparseFeature(name="matrix")
    assert feature.is_placeholder()


def test_sparse_factory_alias_is_matrix_factory() -> None:
    assert Sparse is Matrix


def test_sparse_factory_alias_creates_matrix_feature(pack_dir: Path) -> None:
    feature = Sparse(name="matrix", path=pack_dir)
    assert isinstance(feature, MatrixFeature)
