"""Tests for FlexibleDataset with in-memory .value attribute support.

This module tests the integration of raw arrays via DataEntry.value,
enabling testing without file I/O and supporting programmatic API usage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict, TensorDictBase

from dlkit.runtime.data.datasets.flexible import (
    BatchComplianceError,
    FlexibleDataset,
    PlaceholderNotResolvedError,
    _load_or_convert_tensor,
    _normalize_entries,
)
from dlkit.tools.config.data_entries import (
    Feature,
    PathFeature,
    PathTarget,
    Target,
    ValueFeature,
    ValueTarget,
)


def _expect_tensor(value: object) -> torch.Tensor:
    assert isinstance(value, torch.Tensor)
    return value


def _expect_tensordict(value: object) -> TensorDictBase:
    assert isinstance(value, TensorDictBase)
    return value


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
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) == 1
        assert len(targets.keys()) == 1
        assert isinstance(sample["features", "feat1"], torch.Tensor)
        assert isinstance(sample["targets", "target1"], torch.Tensor)
        assert sample["features", "feat1"].shape == (2,)
        assert sample["targets", "target1"].shape == (3,)

    def test_dataset_with_path_entries_only(
        self, path_feature: PathFeature, path_target: PathTarget
    ):
        """Test FlexibleDataset with only path-based entries."""
        ds = FlexibleDataset(features=[path_feature], targets=[path_target])

        assert len(ds) == 5
        sample = ds[0]
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) >= 1
        assert len(targets.keys()) >= 1

    def test_dataset_with_mixed_entries(self, value_feature: ValueFeature, path_target: PathTarget):
        """Test FlexibleDataset with mixed value and path entries."""
        ds = FlexibleDataset(features=[value_feature], targets=[path_target])

        assert len(ds) == 5
        sample = ds[0]
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) >= 1
        assert len(targets.keys()) >= 1
        assert isinstance(sample["features", "feat1"], torch.Tensor)
        assert isinstance(sample["targets", "target_file"], torch.Tensor)

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
        sample = ds[0]
        assert isinstance(sample, TensorDict)
        assert len(_expect_tensordict(sample["features"]).keys()) == 1
        assert len(_expect_tensordict(sample["targets"]).keys()) == 1

    def test_dataset_with_factory_value_entries(self):
        """Test FlexibleDataset with Feature/Target factory value entries."""
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        Y = np.ones((10, 1), dtype=np.float32)

        feat = Feature(name="X", value=X)
        targ = Target(name="Y", value=Y)

        ds = FlexibleDataset(features=[feat], targets=[targ])

        assert len(ds) == 10
        sample = ds[5]

        assert isinstance(sample, TensorDict)
        assert torch.allclose(
            _expect_tensor(sample["features", "X"]), torch.tensor([10.0, 11.0], dtype=torch.float32)
        )
        assert torch.allclose(
            _expect_tensor(sample["targets", "Y"]), torch.tensor([1.0], dtype=torch.float32)
        )

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

        with pytest.raises(BatchComplianceError, match="same first dimension N"):
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
        sample = ds[0]
        assert isinstance(sample, TensorDict)
        assert len(_expect_tensordict(sample["features"]).keys()) >= 1

    def test_only_targets_valid(self):
        """Test FlexibleDataset with only targets."""
        Y = np.ones((5, 1), dtype=np.float32)
        targ = ValueTarget(name="Y", value=Y)

        ds = FlexibleDataset(features=[], targets=[targ])

        assert len(ds) == 5
        sample = ds[0]
        assert isinstance(sample, TensorDict)
        assert len(_expect_tensordict(sample["targets"]).keys()) >= 1


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

        ds = FlexibleDataset(features=[feat1, feat2, feat3], targets=None)

        assert len(ds) == 5
        sample = ds[0]
        features = _expect_tensordict(sample["features"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) == 3
        assert sample["features", "feat1"].shape == (2,)
        assert sample["features", "feat2"].shape == (3,)
        assert sample["features", "feat3"].shape == (4,)

    def test_torch_tensor_values(self):
        """Test FlexibleDataset with torch.Tensor values."""
        X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
        Y = torch.zeros((10, 1), dtype=torch.float32)

        feat = ValueFeature(name="X", value=X)
        targ = ValueTarget(name="Y", value=Y)

        ds = FlexibleDataset(features=[feat], targets=[targ])

        assert len(ds) == 10
        sample = ds[0]

        assert isinstance(sample, TensorDict)
        assert torch.equal(
            _expect_tensor(sample["features", "X"]), torch.tensor([0.0, 1.0], dtype=torch.float32)
        )
        assert torch.equal(
            _expect_tensor(sample["targets", "Y"]), torch.tensor([0.0], dtype=torch.float32)
        )

    def test_flexible_dataset_with_value_factory_entries(self):
        """Test FlexibleDataset with Feature/Target factory value entries."""
        feat = Feature(name="x", value=np.ones((10, 5), dtype=np.float32))
        targ = Target(name="y", value=np.zeros((10, 1), dtype=np.float32))

        dataset = FlexibleDataset(features=[feat], targets=[targ])

        assert len(dataset) == 10

        batch = dataset[0]
        features = _expect_tensordict(batch["features"])
        targets = _expect_tensordict(batch["targets"])
        assert isinstance(batch, TensorDict)
        assert len(features.keys()) >= 1
        assert len(targets.keys()) >= 1
        assert batch["features", "x"].shape == (5,)
        assert batch["targets", "y"].shape == (1,)
        assert torch.allclose(
            _expect_tensor(batch["features", "x"]), torch.ones(5, dtype=torch.float32)
        )
        assert torch.allclose(
            _expect_tensor(batch["targets", "y"]), torch.zeros(1, dtype=torch.float32)
        )

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
        assert isinstance(batch, TensorDict)
        assert torch.allclose(
            _expect_tensor(batch["features", "x"]), torch.full((3,), 2.0, dtype=torch.float32)
        )
        assert torch.allclose(
            _expect_tensor(batch["targets", "y"]), torch.full((1,), 5.0, dtype=torch.float32)
        )

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

        dataset = FlexibleDataset(features=[feat1, feat2, feat3], targets=[targ])

        assert len(dataset) == 8
        sample = dataset[0]
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) == 3 and len(targets.keys()) == 1
        assert sample["features", "x1"].shape == (2,)
        assert sample["features", "x2"].shape == (3,)
        assert sample["features", "x3"].shape == (4,)
        assert sample["targets", "y"].shape == (1,)
        assert torch.allclose(
            _expect_tensor(sample["targets", "y"]), torch.tensor([10.0], dtype=torch.float32)
        )


# ============================================================================
# Batch Compliance Tests
# ============================================================================


@pytest.fixture
def feature_5x2() -> ValueFeature:
    """ValueFeature with shape (5, 2)."""
    return ValueFeature(name="x", value=np.ones((5, 2), dtype=np.float32))


@pytest.fixture
def target_5x1() -> ValueTarget:
    """ValueTarget with shape (5, 1)."""
    return ValueTarget(name="y", value=np.zeros((5, 1), dtype=np.float32))


class TestBatchCompliance:
    """Tests enforcing strict batch-shape invariants on FlexibleDataset."""

    def test_scalar_feature_raises_batch_compliance_error(self, target_5x1: ValueTarget):
        """Scalar (0-D) feature tensor raises BatchComplianceError."""
        scalar_feat = ValueFeature(name="s", value=np.array(1.0, dtype=np.float32))

        with pytest.raises(BatchComplianceError, match="Scalar"):
            FlexibleDataset(features=[scalar_feat], targets=[target_5x1])

    def test_scalar_target_raises_batch_compliance_error(self, feature_5x2: ValueFeature):
        """Scalar (0-D) target tensor raises BatchComplianceError."""
        scalar_targ = ValueTarget(name="s", value=np.array(0.0, dtype=np.float32))

        with pytest.raises(BatchComplianceError, match="Scalar"):
            FlexibleDataset(features=[feature_5x2], targets=[scalar_targ])

    def test_mismatched_n_raises_batch_compliance_error(self):
        """Entries with different first dimensions raise BatchComplianceError."""
        feat = ValueFeature(name="x", value=np.ones((5, 2), dtype=np.float32))
        targ = ValueTarget(name="y", value=np.zeros((7, 1), dtype=np.float32))

        with pytest.raises(BatchComplianceError, match="same first dimension N"):
            FlexibleDataset(features=[feat], targets=[targ])

    def test_mixed_ranks_with_shared_n_pass(self, feature_5x2: ValueFeature):
        """Entries with different ranks but the same first dimension N are valid."""
        feat_3d = ValueFeature(name="x3", value=np.ones((5, 4, 4), dtype=np.float32))
        feat_4d = ValueFeature(name="x4", value=np.ones((5, 2, 3, 2), dtype=np.float32))
        targ = ValueTarget(name="y", value=np.zeros((5, 1), dtype=np.float32))

        ds = FlexibleDataset(features=[feature_5x2, feat_3d, feat_4d], targets=[targ])

        assert len(ds) == 5
        sample = ds[0]
        assert sample["features", "x"].shape == (2,)
        assert sample["features", "x3"].shape == (4, 4)
        assert sample["features", "x4"].shape == (2, 3, 2)

    def test_dataset_td_has_correct_batch_size(
        self, feature_5x2: ValueFeature, target_5x1: ValueTarget
    ):
        """Internal _dataset_td is built with batch_size=[N]."""
        ds = FlexibleDataset(features=[feature_5x2], targets=[target_5x1])

        assert ds._dataset_td.batch_size == torch.Size([5])
        assert _expect_tensordict(ds._dataset_td["features"]).batch_size == torch.Size([5])
        assert _expect_tensordict(ds._dataset_td["targets"]).batch_size == torch.Size([5])

    def test_getitem_returns_unbatched_tensordict(
        self, feature_5x2: ValueFeature, target_5x1: ValueTarget
    ):
        """__getitem__ returns a TensorDict with batch_size=[] (single sample)."""
        ds = FlexibleDataset(features=[feature_5x2], targets=[target_5x1])

        sample = ds[0]

        assert isinstance(sample, TensorDict)
        assert sample.batch_size == torch.Size([])
