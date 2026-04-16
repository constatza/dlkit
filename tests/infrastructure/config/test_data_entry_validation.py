"""Tests for DataEntry validation logic.

Tests the new hierarchical DataEntry architecture:
- PathBasedEntry/ValueBasedEntry base classes
- PathFeature/ValueFeature/PathTarget/ValueTarget concrete classes
- Feature()/Target() factory functions
- Placeholder mode support
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from dlkit.infrastructure.config.data_entries import (
    Feature,
    PathBasedEntry,
    PathFeature,
    PathTarget,
    Target,
    ValueBasedEntry,
    ValueFeature,
    ValueTarget,
    is_feature_entry,
    is_path_based,
    is_target_entry,
    is_value_based,
)
from dlkit.infrastructure.config.dataset_settings import DatasetSettings

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Sample numpy array for testing."""
    return np.ones((10, 5), dtype=np.float32)


@pytest.fixture
def sample_torch_tensor() -> torch.Tensor:
    """Sample torch tensor for testing."""
    return torch.randn(10, 5, dtype=torch.float32)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFeatureFactory:
    """Tests for Feature() factory function."""

    def test_feature_with_path_returns_path_feature(self, tmp_path: Path):
        """Test Feature with path returns PathFeature instance.

        Note: Uses model_construct to bypass path existence validation,
        since we're testing type detection logic, not path validation.
        """
        test_path = tmp_path / "test.npy"
        feat = PathFeature.model_construct(name="test", path=test_path)

        assert isinstance(feat, PathFeature)
        assert isinstance(feat, PathBasedEntry)
        assert feat.has_path()
        assert not feat.has_value()
        assert not feat.is_placeholder()

    def test_feature_with_value_returns_value_feature(self, sample_numpy_array: np.ndarray):
        """Test Feature with value returns ValueFeature instance."""
        feat = Feature(name="test", value=sample_numpy_array)

        assert isinstance(feat, ValueFeature)
        assert isinstance(feat, ValueBasedEntry)
        assert feat.has_value()
        assert not feat.has_path()
        assert not feat.is_placeholder()

    def test_feature_without_path_or_value_returns_placeholder(self):
        """Test Feature without path or value returns placeholder PathFeature."""
        feat = Feature(name="test")

        assert isinstance(feat, PathFeature)
        assert feat.is_placeholder()
        assert not feat.has_path()
        assert not feat.has_value()

    def test_feature_with_path_none_explicit_returns_placeholder(self):
        """Test Feature with explicit path=None returns placeholder."""
        feat = Feature(name="test", path=None)

        assert isinstance(feat, PathFeature)
        assert feat.is_placeholder()

    def test_feature_with_both_path_and_value_raises(
        self, tmp_path: Path, sample_numpy_array: np.ndarray
    ):
        """Test Feature with both path and value raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both 'path' and 'value'"):
            Feature(  # ty: ignore[no-matching-overload]
                name="test", path=tmp_path / "test.npy", value=sample_numpy_array
            )

    def test_feature_string_path_converted_to_path_object(self, tmp_path: Path):
        """Test Feature converts string path to Path object."""
        feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")

        assert isinstance(feat.path, (Path, str))  # Can be either depending on construction
        assert str(feat.path).endswith("test.npy")


class TestTargetFactory:
    """Tests for Target() factory function."""

    def test_target_with_path_returns_path_target(self, tmp_path: Path):
        """Test Target with path returns PathTarget instance."""
        targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")

        assert isinstance(targ, PathTarget)
        assert isinstance(targ, PathBasedEntry)
        assert targ.has_path()
        assert not targ.has_value()
        assert not targ.is_placeholder()

    def test_target_with_value_returns_value_target(self, sample_numpy_array: np.ndarray):
        """Test Target with value returns ValueTarget instance."""
        targ = Target(name="test", value=sample_numpy_array)

        assert isinstance(targ, ValueTarget)
        assert isinstance(targ, ValueBasedEntry)
        assert targ.has_value()
        assert not targ.has_path()
        assert not targ.is_placeholder()

    def test_target_without_path_or_value_returns_placeholder(self):
        """Test Target without path or value returns placeholder PathTarget."""
        targ = Target(name="test")

        assert isinstance(targ, PathTarget)
        assert targ.is_placeholder()

    def test_target_with_both_path_and_value_raises(
        self, tmp_path: Path, sample_numpy_array: np.ndarray
    ):
        """Test Target with both path and value raises ValueError."""
        with pytest.raises(ValueError, match="cannot have both 'path' and 'value'"):
            Target(  # ty: ignore[no-matching-overload]
                name="test", path=tmp_path / "test.npy", value=sample_numpy_array
            )

    def test_target_has_write_attribute(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test Target preserves write attribute."""
        targ_path = PathTarget.model_construct(name="test", path=tmp_path / "test.npy", write=True)
        targ_value = ValueTarget(name="test", value=sample_numpy_array, write=True)

        assert targ_path.write is True
        assert targ_value.write is True


# ============================================================================
# PathFeature/PathTarget Direct Construction Tests
# ============================================================================


class TestPathFeature:
    """Tests for PathFeature direct construction."""

    def test_path_feature_with_valid_path(self, tmp_path: Path):
        """Test PathFeature with valid path."""
        feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")

        assert feat.has_path()
        assert not feat.has_value()
        assert not feat.is_placeholder()

    def test_path_feature_placeholder_mode(self):
        """Test PathFeature in placeholder mode (no path)."""
        feat = PathFeature(name="test")

        assert feat.is_placeholder()
        assert not feat.has_path()
        assert not feat.has_value()

    def test_path_feature_loss_input_default(self, tmp_path: Path):
        """Test PathFeature defaults loss_input to None."""
        feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")

        assert feat.loss_input is None


class TestPathTarget:
    """Tests for PathTarget direct construction."""

    def test_path_target_with_valid_path(self, tmp_path: Path):
        """Test PathTarget with valid path."""
        targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")

        assert targ.has_path()
        assert not targ.has_value()
        assert not targ.is_placeholder()

    def test_path_target_placeholder_mode(self):
        """Test PathTarget in placeholder mode (no path)."""
        targ = PathTarget(name="test")

        assert targ.is_placeholder()
        assert not targ.has_path()
        assert not targ.has_value()

    def test_path_target_loss_input_default(self, tmp_path: Path):
        """Test PathTarget defaults loss_input to None."""
        targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")

        assert targ.loss_input is None

    def test_path_target_write_attribute(self, tmp_path: Path):
        """Test PathTarget has write attribute."""
        targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy", write=True)

        assert targ.write is True


# ============================================================================
# ValueFeature/ValueTarget Direct Construction Tests
# ============================================================================


class TestValueFeature:
    """Tests for ValueFeature direct construction."""

    def test_value_feature_with_numpy_array(self, sample_numpy_array: np.ndarray):
        """Test ValueFeature with numpy array."""
        feat = ValueFeature(name="test", value=sample_numpy_array)

        assert feat.has_value()
        assert not feat.has_path()
        assert not feat.is_placeholder()
        assert isinstance(feat.value, np.ndarray)

    def test_value_feature_with_torch_tensor(self, sample_torch_tensor: torch.Tensor):
        """Test ValueFeature with torch tensor."""
        feat = ValueFeature(name="test", value=sample_torch_tensor)

        assert feat.has_value()
        assert isinstance(feat.value, torch.Tensor)

    def test_value_feature_loss_input_default(self, sample_numpy_array: np.ndarray):
        """Test ValueFeature defaults loss_input to None."""
        feat = ValueFeature(name="test", value=sample_numpy_array)

        assert feat.loss_input is None


class TestValueTarget:
    """Tests for ValueTarget direct construction."""

    def test_value_target_with_numpy_array(self, sample_numpy_array: np.ndarray):
        """Test ValueTarget with numpy array."""
        targ = ValueTarget(name="test", value=sample_numpy_array)

        assert targ.has_value()
        assert not targ.has_path()
        assert not targ.is_placeholder()
        assert isinstance(targ.value, np.ndarray)

    def test_value_target_with_torch_tensor(self, sample_torch_tensor: torch.Tensor):
        """Test ValueTarget with torch tensor."""
        targ = ValueTarget(name="test", value=sample_torch_tensor)

        assert targ.has_value()
        assert isinstance(targ.value, torch.Tensor)

    def test_value_target_loss_input_default(self, sample_numpy_array: np.ndarray):
        """Test ValueTarget defaults loss_input to None."""
        targ = ValueTarget(name="test", value=sample_numpy_array)

        assert targ.loss_input is None

    def test_value_target_write_attribute(self, sample_numpy_array: np.ndarray):
        """Test ValueTarget has write attribute."""
        targ = ValueTarget(name="test", value=sample_numpy_array, write=True)

        assert targ.write is True


# ============================================================================
# Type Guard Tests
# ============================================================================


class TestTypeGuards:
    """Tests for type guard functions."""

    def test_is_feature_entry(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_feature_entry identifies feature types."""
        path_feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")
        value_feat = ValueFeature(name="test", value=sample_numpy_array)
        path_targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")

        assert is_feature_entry(path_feat) is True
        assert is_feature_entry(value_feat) is True
        assert is_feature_entry(path_targ) is False

    def test_is_target_entry(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_target_entry identifies target types."""
        path_targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")
        value_targ = ValueTarget(name="test", value=sample_numpy_array)
        path_feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")

        assert is_target_entry(path_targ) is True
        assert is_target_entry(value_targ) is True
        assert is_target_entry(path_feat) is False

    def test_is_path_based(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_path_based identifies path-based types."""
        path_feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")
        path_targ = PathTarget.model_construct(name="test", path=tmp_path / "test.npy")
        value_feat = ValueFeature(name="test", value=sample_numpy_array)

        assert is_path_based(path_feat) is True
        assert is_path_based(path_targ) is True
        assert is_path_based(value_feat) is False

    def test_is_value_based(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_value_based identifies value-based types."""
        value_feat = ValueFeature(name="test", value=sample_numpy_array)
        value_targ = ValueTarget(name="test", value=sample_numpy_array)
        path_feat = PathFeature.model_construct(name="test", path=tmp_path / "test.npy")

        assert is_value_based(value_feat) is True
        assert is_value_based(value_targ) is True
        assert is_value_based(path_feat) is False


# ============================================================================
# Strict Validation Mode Tests
# ============================================================================


class TestStrictValidation:
    """Tests for strict validation mode."""

    def test_path_feature_strict_mode_valid_path(self, tmp_path: Path):
        """Test PathFeature passes strict validation with existing path."""
        test_file = tmp_path / "test.npy"
        np.save(test_file, np.ones((10, 5)))

        feat = PathFeature.model_validate(
            {"name": "test", "path": str(test_file)}, context={"strict": True}
        )

        assert feat.has_path()

    def test_path_feature_strict_mode_invalid_path(self):
        """Test PathFeature fails strict validation with non-existing path."""
        with pytest.raises(ValueError, match="Path does not exist"):
            PathFeature.model_validate(
                {"name": "test", "path": "/nonexistent/path.npy"}, context={"strict": True}
            )

    def test_path_feature_non_strict_mode_invalid_path(self):
        """Test PathFeature constructed with non-strict mode allows non-existing path."""
        # Use model_construct to create without validation
        feat = PathFeature.model_construct(name="test", path="/nonexistent/path.npy")

        assert feat.has_path()

    def test_placeholder_strict_mode_ok(self):
        """Test placeholder passes strict validation (no path to check)."""
        feat = PathFeature.model_validate({"name": "test"}, context={"strict": True})

        assert feat.is_placeholder()


# ============================================================================
# Error Message Tests
# ============================================================================


class TestErrorMessages:
    """Tests for error message quality."""

    def test_factory_error_includes_entry_name(
        self, tmp_path: Path, sample_numpy_array: np.ndarray
    ):
        """Test factory error messages include entry name."""
        with pytest.raises(ValueError, match="Feature 'my_feature'"):
            Feature(  # ty: ignore[no-matching-overload]
                name="my_feature", path=tmp_path / "test.npy", value=sample_numpy_array
            )

        with pytest.raises(ValueError, match="Target 'my_target'"):
            Target(  # ty: ignore[no-matching-overload]
                name="my_target", path=tmp_path / "test.npy", value=sample_numpy_array
            )


# ============================================================================
# Name Validation Tests (Production Bug Fix)
# ============================================================================


class TestNameValidation:
    """Tests for name validation when data source is present.

    These tests verify the fix for the production bug where TOML configs
    with missing 'name' fields passed validation but failed at runtime.
    """

    def test_path_feature_missing_name_fails(self, tmp_path: Path):
        """PathFeature with path but no name should fail validation."""
        with pytest.raises(ValueError, match="requires 'name'"):
            PathFeature.model_validate({"path": str(tmp_path / "test.npy")})

    def test_path_feature_missing_name_error_message_includes_details(self, tmp_path: Path):
        """Error message should include helpful details about the problem."""
        with pytest.raises(ValueError) as exc_info:
            PathFeature.model_validate({"path": str(tmp_path / "test.npy")})

        error_msg = str(exc_info.value)
        assert "PathFeature" in error_msg
        assert "requires 'name'" in error_msg
        assert "test.npy" in error_msg
        assert "DATASET.features" in error_msg
        assert "Fix:" in error_msg

    def test_path_target_missing_name_fails(self, tmp_path: Path):
        """PathTarget with path but no name should fail validation."""
        with pytest.raises(ValueError, match="requires 'name'"):
            PathTarget.model_validate({"path": str(tmp_path / "test.npy")})

    def test_value_feature_missing_name_fails(self, sample_numpy_array: np.ndarray):
        """ValueFeature with value but no name should fail validation."""
        with pytest.raises(ValueError, match="requires 'name'"):
            ValueFeature.model_validate({"value": sample_numpy_array})

    def test_value_target_missing_name_fails(self, sample_numpy_array: np.ndarray):
        """ValueTarget with value but no name should fail validation."""
        with pytest.raises(ValueError, match="requires 'name'"):
            ValueTarget.model_validate({"value": sample_numpy_array})

    def test_placeholder_without_name_succeeds(self):
        """Placeholder (no path, no value, no name) should succeed."""
        # PathFeature placeholder
        feat = PathFeature.model_validate({})
        assert feat.is_placeholder()
        assert feat.name is None

        # PathTarget placeholder
        targ = PathTarget.model_validate({})
        assert targ.is_placeholder()
        assert targ.name is None

    def test_placeholder_with_name_no_data_succeeds(self):
        """Placeholder with name but no data should succeed (for later injection)."""
        feat = PathFeature.model_validate({"name": "x"})
        assert feat.is_placeholder()
        assert feat.name == "x"
        assert feat.path is None

    def test_valid_path_feature_with_name_succeeds(self, tmp_path: Path):
        """PathFeature with both name and path should succeed (with model_construct)."""
        feat = PathFeature.model_construct(name="x", path=tmp_path / "test.npy")
        assert feat.name == "x"
        assert str(feat.path).endswith("test.npy")
        assert not feat.is_placeholder()

    def test_valid_value_feature_with_name_succeeds(self, sample_numpy_array: np.ndarray):
        """ValueFeature with both name and value should succeed."""
        feat = ValueFeature.model_validate({"name": "x", "value": sample_numpy_array})
        assert feat.name == "x"
        assert feat.has_value()
        assert not feat.is_placeholder()

    def test_factory_function_respects_validation(self, tmp_path: Path):
        """Factory functions should also trigger validation."""
        # This should fail (path without name)
        with pytest.raises(ValueError, match="requires 'name'"):
            PathFeature(path=tmp_path / "test.npy")

        # This should succeed (placeholder without name)
        feat = PathFeature()
        assert feat.is_placeholder()

    def test_model_construct_bypasses_validation(self, tmp_path: Path):
        """model_construct should bypass validation (for internal use)."""
        # This would normally fail validation, but model_construct skips it
        feat = PathFeature.model_construct(path=tmp_path / "test.npy", name=None)
        assert str(feat.path).endswith("test.npy")
        assert feat.name is None
        # Note: This is intentional for internal/testing use


# ============================================================================
# DatasetSettings Integration Tests (Regression for nested_model issue)
# ============================================================================


class TestDatasetSettingsValuePreservation:
    """Regression tests for ValueFeature/ValueTarget preservation in DatasetSettings.

    These tests guard against the bug where nested_model_default_partial_update=True
    caused ValueFeature/ValueTarget objects to be converted to PathFeature/PathTarget
    during DatasetSettings validation (losing the in-memory value field).
    """

    def test_value_feature_preserved_in_dataset_settings(self, sample_numpy_array: np.ndarray):
        """ValueFeature should remain ValueFeature when stored in DatasetSettings.features."""

        feat = Feature(name="x", value=sample_numpy_array)
        assert isinstance(feat, ValueFeature)
        assert feat.has_value()
        feat_id = id(feat)

        # Create DatasetSettings with ValueFeature
        ds_settings = DatasetSettings(features=(feat,))

        # Verify type preservation
        assert isinstance(ds_settings.features[0], ValueFeature)
        assert not isinstance(ds_settings.features[0], PathFeature)

        # Verify value preservation
        assert ds_settings.features[0].has_value()
        assert ds_settings.features[0].value is not None
        assert ds_settings.features[0].value.shape == sample_numpy_array.shape

        # Verify object identity (no deep copy)
        assert id(ds_settings.features[0]) == feat_id
        assert ds_settings.features[0] is feat

    def test_value_target_preserved_in_dataset_settings(self, sample_numpy_array: np.ndarray):
        """ValueTarget should remain ValueTarget when stored in DatasetSettings.targets."""

        targ = Target(name="y", value=sample_numpy_array)
        assert isinstance(targ, ValueTarget)
        assert targ.has_value()
        targ_id = id(targ)

        # Create DatasetSettings with ValueTarget
        ds_settings = DatasetSettings(features=(), targets=(targ,))

        # Verify type preservation
        assert isinstance(ds_settings.targets[0], ValueTarget)
        assert not isinstance(ds_settings.targets[0], PathTarget)

        # Verify value preservation
        assert ds_settings.targets[0].has_value()
        assert ds_settings.targets[0].value is not None
        assert ds_settings.targets[0].value.shape == sample_numpy_array.shape

        # Verify object identity (no deep copy)
        assert id(ds_settings.targets[0]) == targ_id
        assert ds_settings.targets[0] is targ

    def test_mixed_value_and_path_entries_in_dataset_settings(
        self, tmp_path: Path, sample_numpy_array: np.ndarray
    ):
        """DatasetSettings should handle mixed ValueFeature and PathFeature correctly."""

        # Create path-based feature
        x_path = tmp_path / "x.npy"
        np.save(x_path, sample_numpy_array)
        path_feat = Feature(name="x_path", path=x_path)

        # Create value-based feature
        value_feat = Feature(name="x_value", value=sample_numpy_array)

        # Create DatasetSettings with mixed entries
        ds_settings = DatasetSettings(features=(path_feat, value_feat))

        # Verify path feature is PathFeature
        assert isinstance(ds_settings.features[0], PathFeature)
        assert ds_settings.features[0].has_path()
        assert not ds_settings.features[0].has_value()

        # Verify value feature is ValueFeature
        assert isinstance(ds_settings.features[1], ValueFeature)
        assert ds_settings.features[1].has_value()
        assert not ds_settings.features[1].has_path()

        # Verify object identity
        assert ds_settings.features[0] is path_feat
        assert ds_settings.features[1] is value_feat

    def test_value_entries_work_with_flexible_dataset(self, sample_numpy_array: np.ndarray):
        """ValueFeature/ValueTarget should work with FlexibleDataset when passed through DatasetSettings."""
        from dlkit.engine.data.datasets.flexible import FlexibleDataset

        # Create value-based entries
        feat = Feature(name="x", value=sample_numpy_array)
        targ = Target(name="y", value=sample_numpy_array[:, :1])  # First column as target

        # Create DatasetSettings
        ds_settings = DatasetSettings(features=(feat,), targets=(targ,))

        # Create FlexibleDataset using features/targets from DatasetSettings
        # This is the actual usage pattern in BuildFactory
        dataset = FlexibleDataset(features=ds_settings.features, targets=ds_settings.targets)

        # Verify dataset works
        assert len(dataset) == 10
        sample = dataset[0]
        features = cast(object, sample["features"])
        targets = cast(object, sample["targets"])
        assert len(list(cast("dict[str, torch.Tensor]", features).keys())) == 1
        assert len(list(cast("dict[str, torch.Tensor]", targets).keys())) == 1
        assert sample["features", "x"].shape == torch.Size([5])
        assert sample["targets", "y"].shape == torch.Size([1])
