"""PathFeature / zarr dense pack integration tests for FlexibleDataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from dlkit.engine.data.datasets.flexible import FlexibleDataset
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.infrastructure.io.packs import IArrayPackReader


def _expect_float32_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    assert value.dtype == torch.float32
    return value


def test_getitem_returns_dense_tensor(zarr_matrix_pack: dict[str, Any]) -> None:
    """dataset[0]['features']['mat'] must be a (4, 4) float32 dense tensor."""
    pack_path = zarr_matrix_pack["path"]
    dataset = FlexibleDataset(
        features=[Feature(name="mat", path=pack_path)],
        targets=[Target(name="y", value=torch.zeros(3, 1))],
    )

    sample = dataset[0]
    mat = _expect_float32_tensor(sample["features"]["mat"])

    assert mat.shape == (4, 4)
    assert not mat.is_sparse


def test_getitems_returns_batched_dense_tensor(zarr_matrix_pack: dict[str, Any]) -> None:
    """dataset.__getitems__([0, 1, 2])['features']['mat'] must be (3, 4, 4), no .to_dense()."""
    pack_path = zarr_matrix_pack["path"]
    dataset = FlexibleDataset(
        features=[Feature(name="mat", path=pack_path)],
        targets=[Target(name="y", value=torch.zeros(3, 1))],
    )

    batch = dataset.__getitems__([0, 1, 2])
    mat = _expect_float32_tensor(batch["features"]["mat"])

    assert mat.shape == (3, 4, 4)
    assert not mat.is_sparse


def test_n_resolution_from_pack_reader(zarr_matrix_pack: dict[str, Any]) -> None:
    """Dataset with only a zarr PathFeature (no dense entries) must report len == 3."""
    pack_path = zarr_matrix_pack["path"]
    dataset = FlexibleDataset(
        features=[Feature(name="mat", path=pack_path)],
    )

    assert len(dataset) == 3


def test_broadcast_pack_replicates(zarr_broadcast_pack: dict[str, Any]) -> None:
    """1-sample broadcast pack: dataset[[0, 1, 2]] shape is (3, 4, 4) with equal slices."""
    pack_path = zarr_broadcast_pack["path"]
    expected_matrix = zarr_broadcast_pack["matrix"]

    dataset = FlexibleDataset(
        features=[Feature(name="mat", path=pack_path)],
        targets=[Target(name="y", value=torch.zeros(3, 1))],
    )

    batch = dataset.__getitems__([0, 1, 2])
    mat = _expect_float32_tensor(batch["features"]["mat"])

    assert mat.shape == (3, 4, 4)
    expected = torch.from_numpy(np.stack([expected_matrix] * 3))
    assert torch.allclose(mat, expected)


def test_memmap_cache_dir_bypassed_for_zarr_feature(
    zarr_matrix_pack: dict[str, Any],
    npy_target_3x1: Path,
    tmp_path: Path,
) -> None:
    """With memmap_cache_dir set, zarr pack reader must NOT be replaced by a MemoryMappedTensor."""
    pack_path = zarr_matrix_pack["path"]
    cache_dir = tmp_path / "cache"

    # A file-backed target is required so the memmap path can write the cache.
    # The pack entry bypasses the memmap and is kept as-is.
    dataset = FlexibleDataset(
        features=[Feature(name="mat", path=pack_path)],
        targets=[Target(name="y", path=npy_target_3x1)],
        memmap_cache_dir=cache_dir,
    )

    # The pack binding must still be an IArrayPackReader, not a MemoryMappedTensor.
    assert "mat" in dataset._feature_pack_bindings
    assert isinstance(dataset._feature_pack_bindings["mat"], IArrayPackReader)

    # Functional sanity: index still works when memmap path is active.
    mat = _expect_float32_tensor(dataset[0]["features"]["mat"])
    assert mat.shape == (4, 4)


def test_zarr_target_lazy_injection(zarr_matrix_pack: dict[str, Any]) -> None:
    """FlexibleDataset with a zarr Target must return the correct tensor on index.

    Uses the same zarr pack for both feature and target so that sample counts agree.

    Args:
        zarr_matrix_pack: 3-sample 4x4 float32 zarr dense matrix pack.
    """
    pack_path = zarr_matrix_pack["path"]
    dataset = FlexibleDataset(
        features=[Feature(name="x", path=pack_path)],
        targets=[Target(name="y", path=pack_path)],
    )

    sample = dataset[0]
    tensor = _expect_float32_tensor(sample["targets"]["y"])
    assert tensor.shape == (4, 4)


def test_feature_factory_with_zarr_dir(zarr_matrix_pack: dict[str, Any]) -> None:
    """Feature() with a zarr directory builds FlexibleDataset correctly end-to-end.

    Validates that PathFeature auto-detection works from the factory through dataset
    indexing.

    Args:
        zarr_matrix_pack: 3-sample 4x4 float32 zarr dense matrix pack.
    """
    pack_path = zarr_matrix_pack["path"]
    dataset = FlexibleDataset(
        features=[Feature(name="K", path=pack_path)],
    )

    assert len(dataset) == 3
    sample = dataset[0]
    tensor = _expect_float32_tensor(sample["features"]["K"])
    assert tensor.shape == (4, 4)
