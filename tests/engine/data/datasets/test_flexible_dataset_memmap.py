"""Tests for FlexibleDataset memory-mapped cache functionality.

Covers: build, cache reuse, cache invalidation, DataLoader round-trips,
NPZ sources, value-based entry rejection, and regression against in-memory path.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDictBase
from torch.utils.data import DataLoader

from dlkit.engine.data.datasets.flexible import (
    FlexibleDataset,
    _build_memmap_cache,
    collate_tensordict,
)
from dlkit.infrastructure.config.data_entries import Feature, Target

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expect_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def _expect_tensordict(value: object) -> TensorDictBase:
    assert isinstance(value, TensorDictBase)
    return value


def _make_dataset(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> FlexibleDataset:
    """Build a FlexibleDataset with memmap_cache_dir from npy fixtures."""
    return FlexibleDataset(
        features=[Feature(name="x", path=npy_feature_file["path"])],
        targets=[Target(name="y", path=npy_target_file["path"])],
        memmap_cache_dir=memmap_cache_dir,
    )


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


def test_memmap_dataset_correct_length(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """Dataset length equals the number of rows in the source files."""
    ds = _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    assert len(ds) == npy_feature_file["shape"][0]


def test_memmap_getitem_returns_correct_tensordict_shape(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """__getitem__ returns a TensorDict with nested features/targets of expected shapes."""
    ds = _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    sample = ds[0]
    features = _expect_tensordict(sample["features"])
    targets = _expect_tensordict(sample["targets"])
    assert _expect_tensor(features["x"]).shape == torch.Size(npy_feature_file["shape"][1:])
    assert _expect_tensor(targets["y"]).shape == torch.Size(npy_target_file["shape"][1:])


def test_memmap_data_integrity(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """Spot-check values match the original numpy arrays."""
    ds = _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    for idx in [0, 42, 99]:
        sample = ds[idx]
        expected_x = torch.as_tensor(npy_feature_file["data"][idx])
        expected_y = torch.as_tensor(npy_target_file["data"][idx])
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])
        torch.testing.assert_close(_expect_tensor(features["x"]), expected_x)
        torch.testing.assert_close(_expect_tensor(targets["y"]), expected_y)


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------


def test_memmap_cache_created_on_first_run(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """After first construction, .memmap files and metadata exist in the cache dir."""
    _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)

    assert (memmap_cache_dir / "dlkit_fingerprint.txt").exists()
    assert (memmap_cache_dir / "meta.json").exists()
    assert (memmap_cache_dir / "features" / "x.memmap").exists()
    assert (memmap_cache_dir / "targets" / "y.memmap").exists()


def test_memmap_cache_reused_on_second_instantiation(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """Second instantiation reads from cache without calling _build_memmap_cache."""
    _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)

    with patch(
        "dlkit.engine.data.datasets.flexible._build_memmap_cache", wraps=_build_memmap_cache
    ) as mock_build:
        _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
        mock_build.assert_not_called()


def test_memmap_cache_rebuilt_when_source_changes(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """Cache is wiped and rebuilt when source file mtime changes."""
    _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    old_fp = (memmap_cache_dir / "dlkit_fingerprint.txt").read_text()

    # Touch the feature file to change its mtime
    time.sleep(0.01)
    npy_feature_file["path"].touch()

    _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    new_fp = (memmap_cache_dir / "dlkit_fingerprint.txt").read_text()
    assert new_fp != old_fp


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


def test_memmap_dataloader_round_trip(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """Full DataLoader collate with single worker produces correct batch shapes."""
    ds = _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_tensordict)
    batch = next(iter(loader))
    features = _expect_tensordict(batch["features"])
    targets = _expect_tensordict(batch["targets"])
    assert _expect_tensor(features["x"]).shape == torch.Size([16, *npy_feature_file["shape"][1:]])
    assert _expect_tensor(targets["y"]).shape == torch.Size([16, *npy_target_file["shape"][1:]])


def test_memmap_multi_worker_dataloader(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
    memmap_cache_dir: Path,
) -> None:
    """DataLoader with num_workers=2 produces batches without error."""
    ds = _make_dataset(npy_feature_file, npy_target_file, memmap_cache_dir)
    loader = DataLoader(
        ds,
        batch_size=10,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_tensordict,
    )
    try:
        batches = list(loader)
    except PermissionError as exc:
        raise pytest.skip.Exception(
            f"Sandbox does not permit multi-worker transport setup: {exc}"
        ) from exc
    total = sum(b.batch_size[0] for b in batches)
    assert total == len(ds)


# ---------------------------------------------------------------------------
# NPZ source
# ---------------------------------------------------------------------------


def test_memmap_with_npz_source(tmp_path: Path, memmap_cache_dir: Path) -> None:
    """NPZ source files work correctly with memmap_cache_dir."""
    features = np.random.randn(50, 4).astype(np.float32)
    targets = np.random.randn(50, 1).astype(np.float32)
    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, x=features, y=targets)

    ds = FlexibleDataset(
        features=[Feature(name="x", path=npz_path)],
        targets=[Target(name="y", path=npz_path)],
        memmap_cache_dir=memmap_cache_dir,
    )
    assert len(ds) == 50
    sample = ds[0]
    torch.testing.assert_close(
        _expect_tensor(_expect_tensordict(sample["features"])["x"]),
        torch.as_tensor(features[0]),
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_memmap_raises_for_value_based_entry(tmp_path: Path, memmap_cache_dir: Path) -> None:
    """ValueError raised when a ValueBasedEntry is used with memmap_cache_dir."""
    from dlkit.infrastructure.config.data_entries import Feature as ValueFeature

    tensor_data = torch.randn(10, 3)
    with pytest.raises(ValueError, match="not file-backed"):
        FlexibleDataset(
            features=[ValueFeature(name="x", value=tensor_data)],
            memmap_cache_dir=memmap_cache_dir,
        )


# ---------------------------------------------------------------------------
# Regression: in-memory path unchanged
# ---------------------------------------------------------------------------


def test_inmemory_path_unchanged_when_no_cache_dir(
    npy_feature_file: dict[str, Any],
    npy_target_file: dict[str, Any],
) -> None:
    """Without memmap_cache_dir the existing in-memory path works identically."""
    ds = FlexibleDataset(
        features=[Feature(name="x", path=npy_feature_file["path"])],
        targets=[Target(name="y", path=npy_target_file["path"])],
    )
    assert len(ds) == npy_feature_file["shape"][0]
    sample = ds[0]
    torch.testing.assert_close(
        _expect_tensor(_expect_tensordict(sample["features"])["x"]),
        torch.as_tensor(npy_feature_file["data"][0]),
    )
