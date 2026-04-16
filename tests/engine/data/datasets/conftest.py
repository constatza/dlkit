"""Fixtures for dataset tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dlkit.infrastructure.io.sparse import save_sparse_pack


def _dense_matrices_to_coo(
    matrices: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Convert a list of dense matrices to COO sparse pack arrays.

    Args:
        matrices: Dense matrices to convert (all must have the same shape).

    Returns:
        Tuple of ``(indices, values, nnz_ptr, size)``.
    """
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    nnz_ptr = [0]

    for matrix in matrices:
        rows, cols = np.nonzero(matrix)
        vals = matrix[rows, cols]
        row_parts.append(rows.astype(np.int64))
        col_parts.append(cols.astype(np.int64))
        value_parts.append(vals)
        nnz_ptr.append(nnz_ptr[-1] + int(vals.size))

    indices = np.vstack([np.concatenate(row_parts), np.concatenate(col_parts)])
    values = np.concatenate(value_parts)
    ptr = np.asarray(nnz_ptr, dtype=np.int64)
    return indices, values, ptr, matrices[0].shape


@pytest.fixture
def sparse_collation_pack(tmp_path: Path) -> dict[str, Any]:
    """Save a 4-sample COO pack for sparse collation tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Dict with ``path`` (pack dir) and ``matrices`` (list of dense arrays).
    """
    matrices = [
        np.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]], dtype=np.float64),
        np.array([[5.0, 0.0, 0.0], [0.0, 6.0, 2.0], [0.0, 2.0, 7.0]], dtype=np.float64),
        np.array([[8.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.float64),
        np.array([[11.0, 1.0, 0.0], [1.0, 12.0, 0.0], [0.0, 0.0, 13.0]], dtype=np.float64),
    ]
    indices, values, nnz_ptr, size = _dense_matrices_to_coo(matrices)
    pack_path = tmp_path / "matrix_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)
    return {"path": pack_path, "matrices": matrices}


@pytest.fixture
def sparse_shared_pack(tmp_path: Path) -> dict[str, Any]:
    """Save a 1-sample (broadcast) COO pack for shared-matrix tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Dict with ``path`` (pack dir) and ``matrix`` (the shared dense array).
    """
    shared = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float64)
    indices, values, nnz_ptr, size = _dense_matrices_to_coo([shared])
    pack_path = tmp_path / "shared_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)
    return {"path": pack_path, "matrix": shared}


@pytest.fixture
def sparse_scaled_pack(tmp_path: Path) -> dict[str, Any]:
    """Save a 2-sample COO pack with value_scale=10.0 for denorm tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Dict with ``path``, ``matrices``, and ``scale``.
    """
    matrices = [
        np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.array([[3.0, 1.0], [1.0, 4.0]], dtype=np.float64),
    ]
    indices, values, nnz_ptr, size = _dense_matrices_to_coo(matrices)
    pack_path = tmp_path / "scaled_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size, value_scale=10.0)
    return {"path": pack_path, "matrices": matrices, "scale": 10.0}


@pytest.fixture
def sparse_path_feature_pack(tmp_path: Path) -> dict[str, Any]:
    """Save a 2-sample COO pack for path-based feature auto-detection tests.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Dict with ``path`` and ``matrices``.
    """
    matrices = [
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64),
        np.array([[4.0, 1.0], [1.0, 5.0]], dtype=np.float64),
    ]
    indices, values, nnz_ptr, size = _dense_matrices_to_coo(matrices)
    pack_path = tmp_path / "path_feature_pack"
    save_sparse_pack(pack_path, indices, values, nnz_ptr, size)
    return {"path": pack_path, "matrices": matrices}


@pytest.fixture
def small_npy_feature_file(tmp_path: Path) -> dict[str, Any]:
    """Create a (4, 8) float32 .npy feature file matching sparse_collation_pack (4 samples).

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
    """Create a (4, 1) float32 .npy target file matching sparse_collation_pack (4 samples).

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
