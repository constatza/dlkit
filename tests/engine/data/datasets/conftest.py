"""Fixtures for dataset tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlkit.infrastructure.io.packs import save_array_pack


@pytest.fixture
def small_npy_feature_file(tmp_path: Path) -> dict[str, Any]:
    """Create a (4, 8) float32 .npy feature file with 4 samples.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dictionary with ``path``, ``data`` (ndarray), and ``shape``.
    """
    data = np.random.default_rng(0).random((4, 8)).astype(np.float32)
    path = tmp_path / "small_x.npy"
    np.save(path, data)
    return {"path": path, "data": data, "shape": (4, 8)}


@pytest.fixture
def small_npy_target_file(tmp_path: Path) -> dict[str, Any]:
    """Create a (4, 1) float32 .npy target file with 4 samples.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dictionary with ``path``, ``data`` (ndarray), and ``shape``.
    """
    data = np.random.default_rng(1).random((4, 1)).astype(np.float32)
    path = tmp_path / "small_y.npy"
    np.save(path, data)
    return {"path": path, "data": data, "shape": (4, 1)}


@pytest.fixture
def npy_feature_file(tmp_path: Path) -> dict[str, Any]:
    """Create a .npy feature file for memmap tests.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dictionary with ``path``, ``data`` (ndarray), and ``shape``.
    """
    data = np.random.randn(100, 8).astype(np.float32)
    path = tmp_path / "features.npy"
    np.save(path, data)
    return {"path": path, "data": data, "shape": data.shape}


@pytest.fixture
def npy_target_file(tmp_path: Path) -> dict[str, Any]:
    """Create a .npy target file for memmap tests.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dictionary with ``path``, ``data`` (ndarray), and ``shape``.
    """
    data = np.random.randn(100, 1).astype(np.float32)
    path = tmp_path / "targets.npy"
    np.save(path, data)
    return {"path": path, "data": data, "shape": data.shape}


@pytest.fixture
def memmap_cache_dir(tmp_path: Path) -> Path:
    """Provide a fresh cache directory for memmap tests.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to ``tmp_path / "cache"`` (not yet created).
    """
    return tmp_path / "cache"


@pytest.fixture
def npz_single_array(tmp_path: Path) -> dict[str, Any]:
    """Create NPZ file with single array for auto-detection testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary with file path and expected array data.
    """
    data = np.ones((10, 5), dtype=np.float32)
    path = tmp_path / "single.npz"
    np.savez(path, data=data)

    return {"path": path, "array": data, "key": "data"}


@pytest.fixture
def npz_multi_array(tmp_path: Path) -> dict[str, Any]:
    """Create NPZ file with multiple arrays for key selection testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Dictionary with file path, array keys, and expected array data.
    """
    features = np.random.randn(10, 5).astype(np.float32)
    targets = np.random.randint(0, 2, (10, 1)).astype(np.int64)
    latent = np.zeros((10, 3), dtype=np.float32)

    path = tmp_path / "multi.npz"
    np.savez(path, features=features, targets=targets, latent=latent)

    return {
        "path": path,
        "features": features,
        "targets": targets,
        "latent": latent,
        "keys": ["features", "targets", "latent"],
    }


@pytest.fixture
def npz_empty(tmp_path: Path) -> Path:
    """Create empty NPZ file for edge case testing.

    Args:
        tmp_path: pytest temporary directory fixture

    Returns:
        Path to empty NPZ file.
    """
    path = tmp_path / "empty.npz"
    np.savez(path)
    return path


@pytest.fixture
def npy_target_3x1(tmp_path: Path) -> Path:
    """Numpy target file with 3 samples of shape (1,).

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Path to the saved .npy file.
    """
    path = tmp_path / "target_3x1.npy"
    np.save(path, np.zeros((3, 1), dtype=np.float32))
    return path


@pytest.fixture
def zarr_matrix_pack(tmp_path: Path) -> dict[str, Any]:
    """3-sample 4x4 float32 zarr dense matrix pack.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dict with ``path`` (pack dir) and ``matrices`` (list of dense float32 arrays).
    """
    rng = np.random.default_rng(42)
    matrices = [rng.random((4, 4)).astype(np.float32) for _ in range(3)]
    data = np.stack(matrices, axis=0)
    pack_path = tmp_path / "zarr_matrix_pack"
    save_array_pack(pack_path, data)
    return {"path": pack_path, "matrices": matrices}


@pytest.fixture
def zarr_broadcast_pack(tmp_path: Path) -> dict[str, Any]:
    """1-sample (broadcast) 4x4 float32 zarr dense matrix pack.

    Args:
        tmp_path: pytest temporary directory fixture.

    Returns:
        Dict with ``path`` (pack dir) and ``matrix`` (the single shared float32 array).
    """
    rng = np.random.default_rng(7)
    matrix = rng.random((4, 4)).astype(np.float32)
    data = matrix[np.newaxis]
    pack_path = tmp_path / "zarr_broadcast_pack"
    save_array_pack(pack_path, data)
    return {"path": pack_path, "matrix": matrix}
