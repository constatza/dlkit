"""Tests for SparseFeature data entry validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlkit.infrastructure.config.data_entries import SparseFeature, SparseFilesConfig
from dlkit.infrastructure.io.sparse import PackFiles, save_sparse_pack


@pytest.fixture(name="minimal_sparse_pack")
def minimal_sparse_pack_fixture(tmp_path: Path) -> Path:
    """Write a minimal 2×2 sparse pack and return its directory path.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Path to the saved sparse pack directory.
    """
    pack_path = tmp_path / "pack"
    indices = np.array([[0, 1], [0, 1]], dtype=np.int64)
    values = np.array([1.0, 2.0], dtype=np.float64)
    nnz_ptr = np.array([0, 2], dtype=np.int64)
    save_sparse_pack(pack_path, indices, values, nnz_ptr, (2, 2))
    return pack_path


def test_sparse_feature_validates_payload_directory(minimal_sparse_pack: Path) -> None:
    feature = SparseFeature(name="matrix", path=minimal_sparse_pack)
    assert feature.path == minimal_sparse_pack


def test_sparse_feature_raises_without_payload_files(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_pack"
    bad_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="Expected payload files"):
        SparseFeature(name="matrix", path=bad_path)


def test_sparse_feature_supports_custom_payload_names(tmp_path: Path) -> None:
    pack_path = tmp_path / "custom_pack"
    files = PackFiles(indices="i.npy", values="v.npy", nnz_ptr="p.npy", values_scale="s.npy")
    indices = np.array([[0], [0]], dtype=np.int64)
    values = np.array([1.0], dtype=np.float64)
    nnz_ptr = np.array([0, 1], dtype=np.int64)
    save_sparse_pack(pack_path, indices, values, nnz_ptr, (1, 1), files=files)

    feature = SparseFeature(
        name="matrix",
        path=pack_path,
        files=SparseFilesConfig(
            indices="i.npy", values="v.npy", nnz_ptr="p.npy", values_scale="s.npy"
        ),
        denormalize=True,
    )
    assert feature.files.indices == "i.npy"
    assert feature.denormalize is True


def test_sparse_feature_allows_legacy_pack_without_values_scale(minimal_sparse_pack: Path) -> None:
    (minimal_sparse_pack / "values_scale.npy").unlink()

    feature = SparseFeature(name="matrix", path=minimal_sparse_pack)

    assert feature.path == minimal_sparse_pack
