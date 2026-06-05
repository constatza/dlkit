"""Tests for ZarrLazyReader — lazy indexed reader for native zarr arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from dlkit.infrastructure.io.zarr import ZarrLazyReader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def zarr_array(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Create a native zarr array and return its path alongside the source data.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        Tuple of (zarr store path, underlying numpy array of shape (20, 4)).
    """
    p = tmp_path / "test.zarr"
    z = zarr.open_array(str(p), mode="w", shape=(20, 4), chunks=(1, 4), dtype="float32")
    data = np.arange(80, dtype="float32").reshape(20, 4)
    z[:] = data
    return p, data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_getitem_int_returns_correct_sample(zarr_array: tuple[Path, np.ndarray]) -> None:
    """reader[0] returns a tensor that matches data[0].

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, data = zarr_array
    reader = ZarrLazyReader(path)
    result = reader[0]
    assert torch.allclose(result, torch.from_numpy(data[0]))


def test_getitem_list_returns_batch(zarr_array: tuple[Path, np.ndarray]) -> None:
    """reader[[0, 2, 5]] returns a tensor with shape (3, 4).

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, _ = zarr_array
    reader = ZarrLazyReader(path)
    result = reader[[0, 2, 5]]
    assert result.shape == (3, 4)


def test_getitem_slice_returns_batch(zarr_array: tuple[Path, np.ndarray]) -> None:
    """reader[1:4] returns a tensor with shape (3, 4).

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, _ = zarr_array
    reader = ZarrLazyReader(path)
    result = reader[1:4]
    assert result.shape == (3, 4)


def test_n_samples(zarr_array: tuple[Path, np.ndarray]) -> None:
    """reader.n_samples equals 20.

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, _ = zarr_array
    reader = ZarrLazyReader(path)
    assert reader.n_samples == 20


def test_sample_shape(zarr_array: tuple[Path, np.ndarray]) -> None:
    """reader.sample_shape equals (4,).

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, _ = zarr_array
    reader = ZarrLazyReader(path)
    assert reader.sample_shape == (4,)


def test_values_match_source(zarr_array: tuple[Path, np.ndarray]) -> None:
    """Tensor values returned by the reader match the underlying numpy data.

    Args:
        zarr_array: Fixture providing (path, data).
    """
    path, data = zarr_array
    reader = ZarrLazyReader(path)
    indices = [0, 5, 10, 19]
    result = reader[indices]
    expected = torch.from_numpy(data[indices])
    assert torch.allclose(result, expected)
