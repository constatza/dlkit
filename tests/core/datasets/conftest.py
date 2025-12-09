"""Fixtures for dataset tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest


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
