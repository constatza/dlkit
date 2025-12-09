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
    PlaceholderNotResolvedError,
    _load_or_convert_tensor,
    _normalize_entries,
)
from dlkit.tools.config.data_entries import (
    Feature,
    Target,
    PathFeature,
    PathTarget,
    ValueFeature,
    ValueTarget,
)


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
def value_feature(sample_numpy_array: np.ndarray) -> ValueFeature:
    """ValueFeature with in-memory value."""
    return ValueFeature(name="feat1", value=sample_numpy_array)


@pytest.fixture
def value_target(sample_torch_tensor: torch.Tensor) -> ValueTarget:
    """ValueTarget with in-memory value."""
    return ValueTarget(name="target1", value=sample_torch_tensor)


@pytest.fixture
def path_feature(tmp_path: Path) -> PathFeature:
    """PathFeature with file path."""
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    path = tmp_path / "feature.npy"
    np.save(path, X)
    return PathFeature(name="feat_file", path=path)


@pytest.fixture
def path_target(tmp_path: Path) -> PathTarget:
    """PathTarget with file path."""
    Y = np.ones((5, 1), dtype=np.float32)
    path = tmp_path / "target.npy"
    np.save(path, Y)
    return PathTarget(name="target_file", path=path)


@pytest.fixture
def placeholder_feature() -> PathFeature:
    """PathFeature placeholder (no path)."""
    return PathFeature(name="placeholder_feat")


# ============================================================================
# Tests for _normalize_entries() with new hierarchy
# ============================================================================


class TestNormalizeEntriesNewHierarchy:
    """Tests for _normalize_entries() with new PathBasedEntry/ValueBasedEntry hierarchy."""

    def test_normalize_value_feature(self, value_feature: ValueFeature):
        """Test _normalize_entries() extracts value from ValueFeature."""
        result = _normalize_entries([value_feature])

        assert "feat1" in result
        source, name = result["feat1"]
        assert isinstance(source, np.ndarray)
        assert source.shape == (5, 2)
        assert name == "feat1"

    def test_normalize_path_feature(self, path_feature: PathFeature):
        """Test _normalize_entries() extracts path from PathFeature."""
        result = _normalize_entries([path_feature])

        assert "feat_file" in result
        source, name = result["feat_file"]
        assert isinstance(source, Path)
        assert name == "feat_file"

    def test_normalize_value_target(self, value_target: ValueTarget):
        """Test _normalize_entries() extracts value from ValueTarget."""
        result = _normalize_entries([value_target])

        assert "target1" in result
        source, name = result["target1"]
        assert isinstance(source, torch.Tensor)
        assert source.shape == (5, 3)
        assert name == "target1"

    def test_normalize_path_target(self, path_target: PathTarget):
        """Test _normalize_entries() extracts path from PathTarget."""
        result = _normalize_entries([path_target])

        assert "target_file" in result
        source, name = result["target_file"]
        assert isinstance(source, Path)
        assert name == "target_file"

    def test_normalize_placeholder_raises_error(self, placeholder_feature: PathFeature):
        """Test _normalize_entries() raises error for placeholder entries."""
        with pytest.raises(PlaceholderNotResolvedError, match="placeholder without path or value"):
            _normalize_entries([placeholder_feature])

    def test_normalize_mixed_value_and_path(
        self, value_feature: ValueFeature, path_target: PathTarget
    ):
        """Test _normalize_entries() handles mixed value and path entries."""
        result = _normalize_entries([value_feature, path_target])

        assert len(result) == 2
        source1, name1 = result["feat1"]
        assert isinstance(source1, np.ndarray)
        assert name1 == "feat1"
        source2, name2 = result["target_file"]
        assert isinstance(source2, Path)
        assert name2 == "target_file"


class TestNormalizeEntriesFactoryValidation:
    """Tests for _normalize_entries() with factory-created entries."""

    def test_normalize_rejects_dict_input(self):
        """Test _normalize_entries() rejects raw dict input."""
        with pytest.raises(TypeError, match="no longer accepts raw dicts"):
            _normalize_entries({"feat": "/path/to/file.npy"})

    def test_normalize_rejects_dict_in_list(self):
        """Test _normalize_entries() rejects dict in list."""
        with pytest.raises(TypeError, match="no longer accepts raw dicts"):
            _normalize_entries([{"name": "feat", "path": "/path/to/file.npy"}])

    def test_normalize_empty_returns_empty(self):
        """Test _normalize_entries() returns empty dict for None input."""
        result = _normalize_entries(None)
        assert result == {}

    def test_normalize_factory_created_entries(self, tmp_path: Path):
        """Test _normalize_entries() handles Feature()/Target() factory output."""
        # Create actual file for PathFeature
        x_path = tmp_path / "x.npy"
        np.save(x_path, np.ones((5, 2), dtype=np.float32))

        feat = Feature(name="x", path=x_path)
        targ = Target(name="y", value=np.ones((5, 1)))

        result = _normalize_entries([feat, targ])

        assert len(result) == 2
        source_x, name_x = result["x"]
        assert isinstance(source_x, Path)
        assert name_x == "x"
        source_y, name_y = result["y"]
        assert isinstance(source_y, np.ndarray)
        assert name_y == "y"




# ============================================================================
# Tests for _load_or_convert_tensor()
# ============================================================================


class TestLoadOrConvertTensor:
    """Tests for _load_or_convert_tensor() function."""

    def test_convert_numpy_array(self, sample_numpy_array: np.ndarray):
        """Test _load_or_convert_tensor() converts numpy array to tensor."""
        result = _load_or_convert_tensor(sample_numpy_array)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 2)
        assert result.dtype == torch.float32

    def test_convert_torch_tensor(self, sample_torch_tensor: torch.Tensor):
        """Test _load_or_convert_tensor() handles torch.Tensor input."""
        result = _load_or_convert_tensor(sample_torch_tensor)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 3)
        assert result.dtype == torch.float32

    def test_convert_with_explicit_dtype(self, sample_numpy_array: np.ndarray):
        """Test _load_or_convert_tensor() respects explicit dtype parameter."""
        result = _load_or_convert_tensor(sample_numpy_array, dtype=torch.float64)

        assert result.dtype == torch.float64
        assert result.shape == (5, 2)

    def test_load_from_file_path(self, tmp_path: Path):
        """Test _load_or_convert_tensor() loads from file path."""
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        path = tmp_path / "test.npy"
        np.save(path, X)

        result = _load_or_convert_tensor(path)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (5, 2)
        assert result.dtype == torch.float32

    def test_zero_copy_for_numpy(self):
        """Test _load_or_convert_tensor() uses zero-copy for numpy arrays."""
        arr = np.ones((3, 3), dtype=np.float32)
        tensor = _load_or_convert_tensor(arr)

        # Verify zero-copy by modifying original array
        arr[0, 0] = 999.0
        assert tensor[0, 0].item() == 999.0


# ============================================================================
# Tests for FlexibleDataset with new hierarchy
# ============================================================================


class TestFlexibleDatasetNewHierarchy:
    """Tests for FlexibleDataset with new PathBasedEntry/ValueBasedEntry hierarchy."""

    def test_dataset_with_value_entries_only(
        self, value_feature: ValueFeature, value_target: ValueTarget
    ):
        """Test FlexibleDataset with only value-based entries."""
        ds = FlexibleDataset(features=[value_feature], targets=[value_target])

        assert len(ds) == 5
        sample = ds[0]

        assert set(sample.keys()) == {"feat1", "target1"}
        assert isinstance(sample["feat1"], torch.Tensor)
        assert isinstance(sample["target1"], torch.Tensor)
        assert sample["feat1"].shape == (2,)
        assert sample["target1"].shape == (3,)

    def test_dataset_with_path_entries_only(
        self, path_feature: PathFeature, path_target: PathTarget
    ):
        """Test FlexibleDataset with only path-based entries."""
        ds = FlexibleDataset(features=[path_feature], targets=[path_target])

        assert len(ds) == 5
        sample = ds[0]

        assert "feat_file" in sample
        assert "target_file" in sample

    def test_dataset_with_mixed_entries(
        self, value_feature: ValueFeature, path_target: PathTarget
    ):
        """Test FlexibleDataset with mixed value and path entries."""
        ds = FlexibleDataset(features=[value_feature], targets=[path_target])

        assert len(ds) == 5
        sample = ds[0]

        assert "feat1" in sample
        assert "target_file" in sample
        assert isinstance(sample["feat1"], torch.Tensor)
        assert isinstance(sample["target_file"], torch.Tensor)

    def test_dataset_with_placeholder_raises(self, placeholder_feature: PathFeature):
        """Test FlexibleDataset raises error for placeholder entries."""
        with pytest.raises(PlaceholderNotResolvedError):
            FlexibleDataset(features=[placeholder_feature], targets=None)


class TestFlexibleDatasetFactoryEntries:
    """Tests for FlexibleDataset with Feature()/Target() factory entries."""

    def test_dataset_with_factory_path_entries(self, tmp_path: Path):
        """Test FlexibleDataset with Feature/Target factory path entries."""
        X_path = tmp_path / "X.npy"
        Y_path = tmp_path / "Y.npy"
        np.save(X_path, np.ones((5, 2), dtype=np.float32))
        np.save(Y_path, np.ones((5, 1), dtype=np.float32))

        feat = Feature(name="X", path=X_path)
        targ = Target(name="Y", path=Y_path)

        ds = FlexibleDataset(features=[feat], targets=[targ])

        assert len(ds) == 5
        assert set(ds[0].keys()) == {"X", "Y"}

    def test_dataset_with_factory_value_entries(self):
        """Test FlexibleDataset with Feature/Target factory value entries."""
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        Y = np.ones((10, 1), dtype=np.float32)

        feat = Feature(name="X", value=X)
        targ = Target(name="Y", value=Y)

        ds = FlexibleDataset(features=[feat], targets=[targ])

        assert len(ds) == 10
        sample = ds[5]

        assert torch.allclose(sample["X"], torch.tensor([10.0, 11.0], dtype=torch.float32))
        assert torch.allclose(sample["Y"], torch.tensor([1.0], dtype=torch.float32))

    def test_dataset_with_factory_placeholder_raises(self):
        """Test FlexibleDataset raises error for factory placeholder entries."""
        feat = Feature(name="X")  # Placeholder

        with pytest.raises(PlaceholderNotResolvedError):
            FlexibleDataset(features=[feat], targets=None)


class TestFlexibleDatasetValidation:
    """Tests for FlexibleDataset validation."""

    def test_validates_consistent_lengths(self):
        """Test FlexibleDataset validates consistent tensor lengths."""
        X = np.ones((5, 2), dtype=np.float32)
        Y = np.ones((3, 1), dtype=np.float32)  # Different length!

        feat = ValueFeature(name="X", value=X)
        targ = ValueTarget(name="Y", value=Y)

        with pytest.raises(ValueError, match="same first dimension"):
            FlexibleDataset(features=[feat], targets=[targ])

    def test_requires_at_least_one_entry(self):
        """Test FlexibleDataset requires at least one entry."""
        with pytest.raises(ValueError, match="At least one feature or target"):
            FlexibleDataset(features=[], targets=None)

    def test_only_features_valid(self):
        """Test FlexibleDataset with only features."""
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        feat = ValueFeature(name="X", value=X)

        ds = FlexibleDataset(features=[feat], targets=None)

        assert len(ds) == 5
        assert "X" in ds[0]

    def test_only_targets_valid(self):
        """Test FlexibleDataset with only targets."""
        Y = np.ones((5, 1), dtype=np.float32)
        targ = ValueTarget(name="Y", value=Y)

        ds = FlexibleDataset(features=[], targets=[targ])

        assert len(ds) == 5
        assert "Y" in ds[0]


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for FlexibleDataset with various entry types."""

    def test_all_entry_types_together(self, tmp_path: Path):
        """Test FlexibleDataset with PathFeature and ValueFeature using factories."""
        # PathFeature via factory
        X1_path = tmp_path / "X1.npy"
        np.save(X1_path, np.ones((5, 2), dtype=np.float32))
        feat1 = Feature(name="feat1", path=X1_path)

        # PathFeature direct
        X2_path = tmp_path / "X2.npy"
        np.save(X2_path, np.ones((5, 3), dtype=np.float32) * 2)
        feat2 = PathFeature(name="feat2", path=X2_path)

        # ValueFeature
        X3 = np.ones((5, 4), dtype=np.float32) * 3
        feat3 = ValueFeature(name="feat3", value=X3)

        ds = FlexibleDataset(
            features=[feat1, feat2, feat3],
            targets=None
        )

        assert len(ds) == 5
        sample = ds[0]

        assert set(sample.keys()) == {"feat1", "feat2", "feat3"}
        assert sample["feat1"].shape == (2,)
        assert sample["feat2"].shape == (3,)
        assert sample["feat3"].shape == (4,)

    def test_torch_tensor_values(self):
        """Test FlexibleDataset with torch.Tensor values."""
        X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        Y = torch.zeros((10, 1), dtype=torch.float32)

        feat = ValueFeature(name="X", value=X)
        targ = ValueTarget(name="Y", value=Y)

        ds = FlexibleDataset(features=[feat], targets=[targ])

        assert len(ds) == 10
        sample = ds[0]

        assert torch.equal(sample["X"], torch.tensor([0.0, 1.0], dtype=torch.float32))
        assert torch.equal(sample["Y"], torch.tensor([0.0], dtype=torch.float32))

    def test_flexible_dataset_with_value_factory_entries(self):
        """Test FlexibleDataset with Feature/Target factory value entries."""
        feat = Feature(name="x", value=np.ones((10, 5), dtype=np.float32))
        targ = Target(name="y", value=np.zeros((10, 1), dtype=np.float32))

        dataset = FlexibleDataset(features=[feat], targets=[targ])

        assert len(dataset) == 10

        batch = dataset[0]
        assert "x" in batch
        assert "y" in batch
        assert batch["x"].shape == (5,)
        assert batch["y"].shape == (1,)
        assert torch.allclose(batch["x"], torch.ones(5, dtype=torch.float32))
        assert torch.allclose(batch["y"], torch.zeros(1, dtype=torch.float32))

    def test_flexible_dataset_mixed_factory_types(self, tmp_path: Path):
        """Test FlexibleDataset with mixed factory entries (path and value)."""
        # Create path-based feature
        x_path = tmp_path / "x.npy"
        np.save(x_path, np.full((10, 3), 2.0, dtype=np.float32))

        # Create value-based target
        y_arr = np.full((10, 1), 5.0, dtype=np.float32)

        feat = Feature(name="x", path=x_path)
        targ = Target(name="y", value=y_arr)

        dataset = FlexibleDataset(features=[feat], targets=[targ])

        assert len(dataset) == 10

        batch = dataset[5]
        assert torch.allclose(batch["x"], torch.full((3,), 2.0, dtype=torch.float32))
        assert torch.allclose(batch["y"], torch.full((1,), 5.0, dtype=torch.float32))

    def test_flexible_dataset_all_entry_formats_factory_based(self, tmp_path: Path):
        """Test FlexibleDataset with all supported entry formats using factories."""
        # PathFeature direct
        x1_path = tmp_path / "x1.npy"
        np.save(x1_path, np.ones((8, 2), dtype=np.float32))
        feat1 = PathFeature(name="x1", path=x1_path)

        # ValueFeature
        feat2 = ValueFeature(name="x2", value=np.ones((8, 3), dtype=np.float32) * 2)

        # Feature factory with path
        x3_path = tmp_path / "x3.npy"
        np.save(x3_path, np.ones((8, 4), dtype=np.float32) * 3)
        feat3 = Feature(name="x3", path=x3_path)

        # Target factory with value
        y_arr = np.ones((8, 1), dtype=np.float32) * 10
        targ = Target(name="y", value=y_arr)

        dataset = FlexibleDataset(
            features=[feat1, feat2, feat3],
            targets=[targ]
        )

        assert len(dataset) == 8
        sample = dataset[0]

        assert set(sample.keys()) == {"x1", "x2", "x3", "y"}
        assert sample["x1"].shape == (2,)
        assert sample["x2"].shape == (3,)
        assert sample["x3"].shape == (4,)
        assert sample["y"].shape == (1,)
        assert torch.allclose(sample["y"], torch.tensor([10.0], dtype=torch.float32))
