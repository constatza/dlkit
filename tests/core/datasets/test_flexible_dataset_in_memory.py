"""Tests for FlexibleDataset with in-memory .value attribute support.

This module tests the integration of raw arrays via DataEntry.value,
enabling testing without file I/O and supporting programmatic API usage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.core.datasets.flexible import (
    FlexibleDataset,
    _load_or_convert_tensor,
    _normalize_entries,
)
from dlkit.tools.config.data_entries import Feature, Target


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Sample numpy array for testing."""
    return np.arange(10, dtype=np.float32).reshape(5, 2)


@pytest.fixture
def sample_torch_tensor() -> torch.Tensor:
    """Sample torch tensor for testing."""
    return torch.ones((5, 3), dtype=torch.float32)


@pytest.fixture
def feature_with_value(sample_numpy_array: np.ndarray) -> Feature:
    """Feature with in-memory .value attribute."""
    return Feature(name="feat1", value=sample_numpy_array)


@pytest.fixture
def target_with_value(sample_torch_tensor: torch.Tensor) -> Target:
    """Target with in-memory .value attribute."""
    return Target(name="target1", value=sample_torch_tensor)


@pytest.fixture
def feature_with_path(tmp_path: Path) -> Feature:
    """Feature with file path."""
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    path = tmp_path / "feature.npy"
    np.save(path, X)
    return Feature(name="feat_file", path=path)


@pytest.fixture
def target_with_path(tmp_path: Path) -> Target:
    """Target with file path."""
    Y = np.ones((5, 1), dtype=np.float32)
    path = tmp_path / "target.npy"
    np.save(path, Y)
    return Target(name="target_file", path=path)


# ============================================================================
# Tests for _normalize_entries() with .value
# ============================================================================


def test_normalize_entries_with_value_attribute(feature_with_value: Feature):
    """Test _normalize_entries() extracts .value from DataEntry."""
    result = _normalize_entries([feature_with_value])

    assert "feat1" in result
    assert isinstance(result["feat1"], np.ndarray)
    assert result["feat1"].shape == (5, 2)


def test_normalize_entries_with_path_attribute(feature_with_path: Feature):
    """Test _normalize_entries() extracts .path from DataEntry."""
    result = _normalize_entries([feature_with_path])

    assert "feat_file" in result
    assert isinstance(result["feat_file"], Path)


def test_normalize_entries_mixed_value_and_path(
    feature_with_value: Feature, target_with_path: Target
):
    """Test _normalize_entries() handles mixed value and path entries."""
    result = _normalize_entries([feature_with_value, target_with_path])

    assert len(result) == 2
    assert isinstance(result["feat1"], np.ndarray)
    assert isinstance(result["target_file"], Path)


def test_normalize_entries_torch_tensor_value(target_with_value: Target):
    """Test _normalize_entries() handles torch.Tensor values."""
    result = _normalize_entries([target_with_value])

    assert "target1" in result
    assert isinstance(result["target1"], torch.Tensor)
    assert result["target1"].shape == (5, 3)


def test_normalize_entries_backwards_compatible_dict():
    """Test _normalize_entries() still handles dict input."""
    result = _normalize_entries({"feat": "/path/to/file.npy"})

    assert "feat" in result
    assert isinstance(result["feat"], Path)
    assert result["feat"] == Path("/path/to/file.npy")


def test_normalize_entries_backwards_compatible_legacy_attrs():
    """Test _normalize_entries() handles legacy name/path attributes."""

    class LegacyEntry:
        def __init__(self):
            self.name = "legacy"
            self.path = "/path/to/legacy.npy"

    result = _normalize_entries([LegacyEntry()])

    assert "legacy" in result
    assert isinstance(result["legacy"], Path)


def test_normalize_entries_empty():
    """Test _normalize_entries() returns empty dict for None input."""
    result = _normalize_entries(None)
    assert result == {}


# ============================================================================
# Tests for _load_or_convert_tensor()
# ============================================================================


def test_load_or_convert_tensor_numpy_array(sample_numpy_array: np.ndarray):
    """Test _load_or_convert_tensor() converts numpy array to tensor."""
    result = _load_or_convert_tensor(sample_numpy_array)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (5, 2)
    assert result.dtype == torch.float32  # Default precision


def test_load_or_convert_tensor_torch_tensor(sample_torch_tensor: torch.Tensor):
    """Test _load_or_convert_tensor() handles torch.Tensor input."""
    result = _load_or_convert_tensor(sample_torch_tensor)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (5, 3)
    assert result.dtype == torch.float32


def test_load_or_convert_tensor_with_explicit_dtype(sample_numpy_array: np.ndarray):
    """Test _load_or_convert_tensor() respects explicit dtype parameter."""
    result = _load_or_convert_tensor(sample_numpy_array, dtype=torch.float64)

    assert result.dtype == torch.float64
    assert result.shape == (5, 2)


def test_load_or_convert_tensor_file_path(tmp_path: Path):
    """Test _load_or_convert_tensor() delegates to load_array() for paths."""
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    path = tmp_path / "test.npy"
    np.save(path, X)

    result = _load_or_convert_tensor(path)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (5, 2)
    assert result.dtype == torch.float32


def test_load_or_convert_tensor_zero_copy_numpy():
    """Test _load_or_convert_tensor() uses zero-copy for numpy arrays."""
    arr = np.ones((3, 3), dtype=np.float32)
    tensor = _load_or_convert_tensor(arr)

    # Verify zero-copy by modifying original array
    arr[0, 0] = 999.0
    assert tensor[0, 0].item() == 999.0


# ============================================================================
# Tests for FlexibleDataset with in-memory values
# ============================================================================


def test_flexible_dataset_with_value_only(
    feature_with_value: Feature, target_with_value: Target
):
    """Test FlexibleDataset with only in-memory .value entries."""
    ds = FlexibleDataset(features=[feature_with_value], targets=[target_with_value])

    assert len(ds) == 5
    sample = ds[0]

    assert set(sample.keys()) == {"feat1", "target1"}
    assert isinstance(sample["feat1"], torch.Tensor)
    assert isinstance(sample["target1"], torch.Tensor)
    assert sample["feat1"].shape == (2,)  # Single row
    assert sample["target1"].shape == (3,)  # Single row


def test_flexible_dataset_mixed_value_and_path(
    feature_with_value: Feature, target_with_path: Target
):
    """Test FlexibleDataset with mixed in-memory and file-based entries."""
    ds = FlexibleDataset(features=[feature_with_value], targets=[target_with_path])

    assert len(ds) == 5
    sample = ds[0]

    assert "feat1" in sample
    assert "target_file" in sample
    assert isinstance(sample["feat1"], torch.Tensor)
    assert isinstance(sample["target_file"], torch.Tensor)


def test_flexible_dataset_with_numpy_value():
    """Test FlexibleDataset with numpy array values."""
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    Y = np.ones((10, 1), dtype=np.float32)

    feat = Feature(name="X", value=X)
    targ = Target(name="Y", value=Y)

    ds = FlexibleDataset(features=[feat], targets=[targ])

    assert len(ds) == 10
    sample = ds[5]

    assert torch.allclose(sample["X"], torch.tensor([10.0, 11.0], dtype=torch.float32))
    assert torch.allclose(sample["Y"], torch.tensor([1.0], dtype=torch.float32))


def test_flexible_dataset_with_torch_tensor_value():
    """Test FlexibleDataset with torch.Tensor values."""
    X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    Y = torch.zeros((10, 1), dtype=torch.float32)

    feat = Feature(name="X", value=X)
    targ = Target(name="Y", value=Y)

    ds = FlexibleDataset(features=[feat], targets=[targ])

    assert len(ds) == 10
    sample = ds[0]

    assert torch.equal(sample["X"], torch.tensor([0.0, 1.0], dtype=torch.float32))
    assert torch.equal(sample["Y"], torch.tensor([0.0], dtype=torch.float32))


def test_flexible_dataset_value_respects_length_validation():
    """Test FlexibleDataset validates consistent lengths for .value entries."""
    X = np.ones((5, 2), dtype=np.float32)
    Y = np.ones((3, 1), dtype=np.float32)  # Different length!

    feat = Feature(name="X", value=X)
    targ = Target(name="Y", value=Y)

    with pytest.raises(ValueError, match="same first dimension"):
        FlexibleDataset(features=[feat], targets=[targ])


def test_flexible_dataset_only_features_with_value():
    """Test FlexibleDataset with only features (no targets) using .value."""
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    feat = Feature(name="X", value=X)

    ds = FlexibleDataset(features=[feat], targets=None)

    assert len(ds) == 5
    sample = ds[0]
    assert "X" in sample
    assert "Y" not in sample


def test_flexible_dataset_only_targets_with_value():
    """Test FlexibleDataset with only targets (no features) using .value."""
    Y = np.ones((5, 1), dtype=np.float32)
    targ = Target(name="Y", value=Y)

    ds = FlexibleDataset(features=[], targets=[targ])

    assert len(ds) == 5
    sample = ds[0]
    assert "Y" in sample


# ============================================================================
# Integration Tests
# ============================================================================


def test_integration_all_three_sources(tmp_path: Path):
    """Test FlexibleDataset with dict paths, Feature.path, and Feature.value."""
    # Dict-based feature (legacy)
    X1_path = tmp_path / "X1.npy"
    np.save(X1_path, np.ones((5, 2), dtype=np.float32))

    # Feature with path
    X2_path = tmp_path / "X2.npy"
    np.save(X2_path, np.ones((5, 3), dtype=np.float32) * 2)
    feat2 = Feature(name="feat2", path=X2_path)

    # Feature with value
    X3 = np.ones((5, 4), dtype=np.float32) * 3
    feat3 = Feature(name="feat3", value=X3)

    ds = FlexibleDataset(
        features=[{"name": "feat1", "path": X1_path}, feat2, feat3], targets=None
    )

    assert len(ds) == 5
    sample = ds[0]

    assert set(sample.keys()) == {"feat1", "feat2", "feat3"}
    assert sample["feat1"].shape == (2,)
    assert sample["feat2"].shape == (3,)
    assert sample["feat3"].shape == (4,)
