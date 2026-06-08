"""Tests for FlexibleDataset with in-memory .value attribute support.

This module tests the integration of raw arrays via DataEntry.value,
enabling testing without file I/O and supporting programmatic API usage.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from tensordict import TensorDict, TensorDictBase

from dlkit.engine.data.datasets.flexible import (
    BatchComplianceError,
    FlexibleDataset,
    PlaceholderNotResolvedError,
    _load_or_convert_tensor,
    _normalize_entries,
    _NormalizedEntry,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import NpyEntry, PathBasedEntry, ValueEntry


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
def value_feature(sample_numpy_array: np.ndarray) -> ValueEntry:
    """ValueEntry with in-memory value (feature role)."""
    return ValueEntry(name="feat1", value=sample_numpy_array, data_role=DataRole.FEATURE)


@pytest.fixture
def value_target(sample_torch_tensor: torch.Tensor) -> ValueEntry:
    """ValueEntry with in-memory value (target role)."""
    return ValueEntry(name="target1", value=sample_torch_tensor, data_role=DataRole.TARGET)


@pytest.fixture
def path_feature(tmp_path: Path) -> NpyEntry:
    """NpyEntry with file path (feature role)."""
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    path = tmp_path / "feature.npy"
    np.save(path, X)
    return NpyEntry(name="feat_file", path=path, data_role=DataRole.FEATURE)


@pytest.fixture
def path_target(tmp_path: Path) -> NpyEntry:
    """NpyEntry with file path (target role)."""
    Y = np.ones((5, 1), dtype=np.float32)
    path = tmp_path / "target.npy"
    np.save(path, Y)
    return NpyEntry(name="target_file", path=path, data_role=DataRole.TARGET)


@pytest.fixture
def placeholder_feature() -> PathBasedEntry:
    """PathBasedEntry placeholder (no path)."""
    from dlkit.infrastructure.config.entry_types import NpyEntry as _NpyEntry

    # NpyEntry without path = placeholder
    return _NpyEntry(name="placeholder_feat", data_role=DataRole.FEATURE)


# ============================================================================
# Tests for _normalize_entries() with new hierarchy
# ============================================================================


class TestNormalizeEntriesNewHierarchy:
    """Tests for _normalize_entries() with new PathBasedEntry/ValueBasedEntry hierarchy."""

    def test_normalize_value_feature(self, value_feature: ValueEntry):
        """Test _normalize_entries() extracts value from ValueEntry (feature)."""
        result = _normalize_entries([value_feature])

        assert "feat1" in result
        ne = result["feat1"]
        assert isinstance(ne, _NormalizedEntry)
        assert isinstance(ne.source, np.ndarray)
        assert ne.source.shape == (5, 2)
        assert ne.array_key == "feat1"

    def test_normalize_path_feature(self, path_feature: NpyEntry):
        """Test _normalize_entries() extracts path from NpyEntry (feature)."""
        result = _normalize_entries([path_feature])

        assert "feat_file" in result
        ne = result["feat_file"]
        assert isinstance(ne, _NormalizedEntry)
        assert isinstance(ne.source, Path)
        assert ne.array_key == "feat_file"

    def test_normalize_value_target(self, value_target: ValueEntry):
        """Test _normalize_entries() extracts value from ValueEntry (target)."""
        result = _normalize_entries([value_target])

        assert "target1" in result
        ne = result["target1"]
        assert isinstance(ne, _NormalizedEntry)
        assert isinstance(ne.source, torch.Tensor)
        assert ne.source.shape == (5, 3)
        assert ne.array_key == "target1"

    def test_normalize_path_target(self, path_target: NpyEntry):
        """Test _normalize_entries() extracts path from NpyEntry (target)."""
        result = _normalize_entries([path_target])

        assert "target_file" in result
        ne = result["target_file"]
        assert isinstance(ne, _NormalizedEntry)
        assert isinstance(ne.source, Path)
        assert ne.array_key == "target_file"

    def test_normalize_placeholder_raises_error(self, placeholder_feature: PathBasedEntry):
        """Test _normalize_entries() raises error for placeholder entries."""
        with pytest.raises(PlaceholderNotResolvedError, match="placeholder without path or value"):
            _normalize_entries([placeholder_feature])

    def test_normalize_mixed_value_and_path(self, value_feature: ValueEntry, path_target: NpyEntry):
        """Test _normalize_entries() handles mixed value and path entries."""
        result = _normalize_entries([value_feature, path_target])

        assert len(result) == 2
        ne1 = result["feat1"]
        assert isinstance(ne1, _NormalizedEntry)
        assert isinstance(ne1.source, np.ndarray)
        assert ne1.array_key == "feat1"
        ne2 = result["target_file"]
        assert isinstance(ne2, _NormalizedEntry)
        assert isinstance(ne2.source, Path)
        assert ne2.array_key == "target_file"


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
        """Test _normalize_entries() handles NpyEntry/ValueEntry factory output."""
        # Create actual file for NpyEntry
        x_path = tmp_path / "x.npy"
        np.save(x_path, np.ones((5, 2), dtype=np.float32))

        feat = NpyEntry(name="x", path=x_path, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="y", value=np.ones((5, 1)), data_role=DataRole.TARGET)

        result = _normalize_entries([feat, targ])

        assert len(result) == 2
        ne_x = result["x"]
        assert isinstance(ne_x, _NormalizedEntry)
        assert isinstance(ne_x.source, Path)
        assert ne_x.array_key == "x"
        ne_y = result["y"]
        assert isinstance(ne_y, _NormalizedEntry)
        assert isinstance(ne_y.source, np.ndarray)
        assert ne_y.array_key == "y"


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

    def test_convert_read_only_numpy_array_copies_before_tensor_conversion(self) -> None:
        """Read-only in-memory arrays must not trigger PyTorch writability warnings."""
        array = np.arange(6, dtype=np.float32).reshape(3, 2)
        array.flags.writeable = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _load_or_convert_tensor(array)

        result[0, 0] = 42.0

        assert result[0, 0].item() == pytest.approx(42.0)
        assert all("not writable" not in str(warning.message) for warning in caught)

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
        self, value_feature: ValueEntry, value_target: ValueEntry
    ):
        """Test FlexibleDataset with only value-based entries."""
        ds = FlexibleDataset(entries=[value_feature, value_target])

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

    def test_dataset_with_path_entries_only(self, path_feature: NpyEntry, path_target: NpyEntry):
        """Test FlexibleDataset with only path-based entries."""
        ds = FlexibleDataset(entries=[path_feature, path_target])

        assert len(ds) == 5
        sample = ds[0]
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) >= 1
        assert len(targets.keys()) >= 1

    def test_dataset_with_mixed_entries(self, value_feature: ValueEntry, path_target: NpyEntry):
        """Test FlexibleDataset with mixed value and path entries."""
        ds = FlexibleDataset(entries=[value_feature, path_target])

        assert len(ds) == 5
        sample = ds[0]
        features = _expect_tensordict(sample["features"])
        targets = _expect_tensordict(sample["targets"])

        assert isinstance(sample, TensorDict)
        assert len(features.keys()) >= 1
        assert len(targets.keys()) >= 1
        assert isinstance(sample["features", "feat1"], torch.Tensor)
        assert isinstance(sample["targets", "target_file"], torch.Tensor)

    def test_dataset_with_placeholder_raises(self, placeholder_feature: PathBasedEntry):
        """Test FlexibleDataset raises error for placeholder entries."""
        with pytest.raises(PlaceholderNotResolvedError):
            FlexibleDataset(entries=cast("Any", [placeholder_feature]))


class TestFlexibleDatasetFactoryEntries:
    """Tests for FlexibleDataset with NpyEntry/ValueEntry entries."""

    def test_dataset_with_factory_path_entries(self, tmp_path: Path):
        """Test FlexibleDataset with NpyEntry path entries."""
        X_path = tmp_path / "X.npy"
        Y_path = tmp_path / "Y.npy"
        np.save(X_path, np.ones((5, 2), dtype=np.float32))
        np.save(Y_path, np.ones((5, 1), dtype=np.float32))

        feat = NpyEntry(name="X", path=X_path, data_role=DataRole.FEATURE)
        targ = NpyEntry(name="Y", path=Y_path, data_role=DataRole.TARGET)

        ds = FlexibleDataset(entries=[feat, targ])

        assert len(ds) == 5
        sample = ds[0]
        assert isinstance(sample, TensorDict)
        assert len(_expect_tensordict(sample["features"]).keys()) == 1
        assert len(_expect_tensordict(sample["targets"]).keys()) == 1

    def test_dataset_with_factory_value_entries(self):
        """Test FlexibleDataset with ValueEntry entries."""
        X = np.arange(20, dtype=np.float32).reshape(10, 2)
        Y = np.ones((10, 1), dtype=np.float32)

        feat = ValueEntry(name="X", value=X, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="Y", value=Y, data_role=DataRole.TARGET)

        ds = FlexibleDataset(entries=[feat, targ])

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
        feat = NpyEntry(name="X", data_role=DataRole.FEATURE)  # Placeholder

        with pytest.raises(PlaceholderNotResolvedError):
            FlexibleDataset(entries=[feat])


class TestFlexibleDatasetValidation:
    """Tests for FlexibleDataset validation."""

    def test_validates_consistent_lengths(self):
        """Test FlexibleDataset validates consistent tensor lengths."""
        X = np.ones((5, 2), dtype=np.float32)
        Y = np.ones((3, 1), dtype=np.float32)  # Different length!

        feat = ValueEntry(name="X", value=X, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="Y", value=Y, data_role=DataRole.TARGET)

        with pytest.raises(BatchComplianceError, match="same first dimension N"):
            FlexibleDataset(entries=[feat, targ])

    def test_requires_at_least_one_entry(self):
        """Test FlexibleDataset requires at least one entry."""
        with pytest.raises(ValueError, match="At least one feature or target"):
            FlexibleDataset(entries=[])

    def test_only_features_valid(self):
        """Test FlexibleDataset with only features."""
        X = np.arange(10, dtype=np.float32).reshape(5, 2)
        feat = ValueEntry(name="X", value=X, data_role=DataRole.FEATURE)

        ds = FlexibleDataset(entries=[feat])

        assert len(ds) == 5
        sample = ds[0]
        assert isinstance(sample, TensorDict)
        assert len(_expect_tensordict(sample["features"]).keys()) >= 1

    def test_only_targets_valid(self):
        """Test FlexibleDataset with only targets."""
        Y = np.ones((5, 1), dtype=np.float32)
        targ = ValueEntry(name="Y", value=Y, data_role=DataRole.TARGET)

        ds = FlexibleDataset(entries=[targ])

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
        """Test FlexibleDataset with NpyEntry and ValueEntry feature entries."""
        # NpyEntry path-based
        X1_path = tmp_path / "X1.npy"
        np.save(X1_path, np.ones((5, 2), dtype=np.float32))
        feat1 = NpyEntry(name="feat1", path=X1_path, data_role=DataRole.FEATURE)

        # NpyEntry direct
        X2_path = tmp_path / "X2.npy"
        np.save(X2_path, np.ones((5, 3), dtype=np.float32) * 2)
        feat2 = NpyEntry(name="feat2", path=X2_path, data_role=DataRole.FEATURE)

        # ValueEntry
        X3 = np.ones((5, 4), dtype=np.float32) * 3
        feat3 = ValueEntry(name="feat3", value=X3, data_role=DataRole.FEATURE)

        ds = FlexibleDataset(entries=[feat1, feat2, feat3])

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

        feat = ValueEntry(name="X", value=X, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="Y", value=Y, data_role=DataRole.TARGET)

        ds = FlexibleDataset(entries=[feat, targ])

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
        """Test FlexibleDataset with ValueEntry entries."""
        feat = ValueEntry(
            name="x", value=np.ones((10, 5), dtype=np.float32), data_role=DataRole.FEATURE
        )
        targ = ValueEntry(
            name="y", value=np.zeros((10, 1), dtype=np.float32), data_role=DataRole.TARGET
        )

        dataset = FlexibleDataset(entries=[feat, targ])

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
        """Test FlexibleDataset with mixed path and value entries."""
        # Create path-based feature
        x_path = tmp_path / "x.npy"
        np.save(x_path, np.full((10, 3), 2.0, dtype=np.float32))

        # Create value-based target
        y_arr = np.full((10, 1), 5.0, dtype=np.float32)

        feat = NpyEntry(name="x", path=x_path, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="y", value=y_arr, data_role=DataRole.TARGET)

        dataset = FlexibleDataset(entries=[feat, targ])

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
        """Test FlexibleDataset with all supported entry formats using new API."""
        # NpyEntry path-based feature
        x1_path = tmp_path / "x1.npy"
        np.save(x1_path, np.ones((8, 2), dtype=np.float32))
        feat1 = NpyEntry(name="x1", path=x1_path, data_role=DataRole.FEATURE)

        # ValueEntry feature
        feat2 = ValueEntry(
            name="x2", value=np.ones((8, 3), dtype=np.float32) * 2, data_role=DataRole.FEATURE
        )

        # NpyEntry feature (second path-based)
        x3_path = tmp_path / "x3.npy"
        np.save(x3_path, np.ones((8, 4), dtype=np.float32) * 3)
        feat3 = NpyEntry(name="x3", path=x3_path, data_role=DataRole.FEATURE)

        # ValueEntry target
        y_arr = np.ones((8, 1), dtype=np.float32) * 10
        targ = ValueEntry(name="y", value=y_arr, data_role=DataRole.TARGET)

        dataset = FlexibleDataset(entries=[feat1, feat2, feat3, targ])

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
def feature_5x2() -> ValueEntry:
    """ValueEntry with shape (5, 2) and feature role."""
    return ValueEntry(name="x", value=np.ones((5, 2), dtype=np.float32), data_role=DataRole.FEATURE)


@pytest.fixture
def target_5x1() -> ValueEntry:
    """ValueEntry with shape (5, 1) and target role."""
    return ValueEntry(name="y", value=np.zeros((5, 1), dtype=np.float32), data_role=DataRole.TARGET)


class TestBatchCompliance:
    """Tests enforcing strict batch-shape invariants on FlexibleDataset."""

    def test_scalar_feature_raises_batch_compliance_error(self, target_5x1: ValueEntry):
        """Scalar (0-D) feature tensor raises BatchComplianceError."""
        scalar_feat = ValueEntry(
            name="s", value=np.array(1.0, dtype=np.float32), data_role=DataRole.FEATURE
        )

        with pytest.raises(BatchComplianceError, match="Scalar"):
            FlexibleDataset(entries=[scalar_feat, target_5x1])

    def test_scalar_target_raises_batch_compliance_error(self, feature_5x2: ValueEntry):
        """Scalar (0-D) target tensor raises BatchComplianceError."""
        scalar_targ = ValueEntry(
            name="s", value=np.array(0.0, dtype=np.float32), data_role=DataRole.TARGET
        )

        with pytest.raises(BatchComplianceError, match="Scalar"):
            FlexibleDataset(entries=[feature_5x2, scalar_targ])

    def test_mismatched_n_raises_batch_compliance_error(self):
        """Entries with different first dimensions raise BatchComplianceError."""
        feat = ValueEntry(
            name="x", value=np.ones((5, 2), dtype=np.float32), data_role=DataRole.FEATURE
        )
        targ = ValueEntry(
            name="y", value=np.zeros((7, 1), dtype=np.float32), data_role=DataRole.TARGET
        )

        with pytest.raises(BatchComplianceError, match="same first dimension N"):
            FlexibleDataset(entries=[feat, targ])

    def test_mixed_ranks_with_shared_n_pass(self, feature_5x2: ValueEntry):
        """Entries with different ranks but the same first dimension N are valid."""
        feat_3d = ValueEntry(
            name="x3", value=np.ones((5, 4, 4), dtype=np.float32), data_role=DataRole.FEATURE
        )
        feat_4d = ValueEntry(
            name="x4", value=np.ones((5, 2, 3, 2), dtype=np.float32), data_role=DataRole.FEATURE
        )
        targ = ValueEntry(
            name="y", value=np.zeros((5, 1), dtype=np.float32), data_role=DataRole.TARGET
        )

        ds = FlexibleDataset(entries=[feature_5x2, feat_3d, feat_4d, targ])

        assert len(ds) == 5
        sample = ds[0]
        assert sample["features", "x"].shape == (2,)
        assert sample["features", "x3"].shape == (4, 4)
        assert sample["features", "x4"].shape == (2, 3, 2)

    def test_dataset_td_has_correct_batch_size(
        self, feature_5x2: ValueEntry, target_5x1: ValueEntry
    ):
        """Internal _dataset_td is built with batch_size=[N]."""
        ds = FlexibleDataset(entries=[feature_5x2, target_5x1])

        assert ds._dataset_td.batch_size == torch.Size([5])
        assert _expect_tensordict(ds._dataset_td["features"]).batch_size == torch.Size([5])
        assert _expect_tensordict(ds._dataset_td["targets"]).batch_size == torch.Size([5])

    def test_getitem_returns_unbatched_tensordict(
        self, feature_5x2: ValueEntry, target_5x1: ValueEntry
    ):
        """__getitem__ returns a TensorDict with batch_size=[] (single sample)."""
        ds = FlexibleDataset(entries=[feature_5x2, target_5x1])

        sample = ds[0]

        assert isinstance(sample, TensorDict)
        assert sample.batch_size == torch.Size([])
