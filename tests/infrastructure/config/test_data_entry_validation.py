"""Tests for DataEntry validation logic.

Tests the new format-specific DataEntry architecture:
- PathBasedEntry/ValueBasedEntry base classes
- NpyEntry/NpzEntry/ZarrEntry/ValueEntry concrete classes
- data_role field (DataRole.FEATURE / DataRole.TARGET)
- Placeholder mode support (path=None)
- is_feature / is_target helpers
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from dlkit.infrastructure.config.data_entries import (
    PathBasedEntry,
    ValueBasedEntry,
    is_feature,
    is_path_based,
    is_target,
    is_value_based,
)
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.dataset_settings import DatasetSettings
from dlkit.infrastructure.config.entry_types import (
    AutoencoderTarget,
    CsvEntry,
    Hdf5Entry,
    NpyEntry,
    NpzEntry,
    ParquetEntry,
    ValueEntry,
    ZarrEntry,
)

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
# NpyEntry Construction Tests
# ============================================================================


class TestNpyEntryFeature:
    """Tests for NpyEntry with feature role."""

    def test_npy_entry_feature_with_valid_path(self, tmp_path: Path):
        """Test NpyEntry (feature) with valid path."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 2)))
        feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)

        assert feat.has_path()
        assert not feat.has_value()
        assert not feat.is_placeholder()
        assert feat.data_role == DataRole.FEATURE

    def test_npy_entry_placeholder_mode(self):
        """Test NpyEntry in placeholder mode (no path)."""
        feat = NpyEntry(name="test", data_role=DataRole.FEATURE)

        assert feat.is_placeholder()
        assert not feat.has_path()
        assert not feat.has_value()

    def test_npy_entry_loss_input_default(self, tmp_path: Path):
        """Test NpyEntry defaults loss_input to None."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 2)))
        feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)

        assert feat.loss_input is None


class TestNpyEntryTarget:
    """Tests for NpyEntry with target role."""

    def test_npy_entry_target_with_valid_path(self, tmp_path: Path):
        """Test NpyEntry (target) with valid path."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 1)))
        targ = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.TARGET)

        assert targ.has_path()
        assert not targ.has_value()
        assert not targ.is_placeholder()
        assert targ.data_role == DataRole.TARGET

    def test_npy_entry_target_placeholder_mode(self):
        """Test NpyEntry (target) in placeholder mode (no path)."""
        targ = NpyEntry(name="test", data_role=DataRole.TARGET)

        assert targ.is_placeholder()
        assert not targ.has_path()
        assert not targ.has_value()

    def test_npy_entry_target_loss_input_default(self, tmp_path: Path):
        """Test NpyEntry (target) defaults loss_input to None."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 1)))
        targ = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.TARGET)

        assert targ.loss_input is None

    def test_npy_entry_target_write_attribute(self, tmp_path: Path):
        """Test NpyEntry (target) has write attribute."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 1)))
        targ = NpyEntry.model_construct(
            name="test", path=path, data_role=DataRole.TARGET, write=True
        )

        assert targ.write is True


# ============================================================================
# ValueEntry Construction Tests
# ============================================================================


class TestValueEntryFeature:
    """Tests for ValueEntry with feature role."""

    def test_value_entry_feature_with_numpy_array(self, sample_numpy_array: np.ndarray):
        """Test ValueEntry (feature) with numpy array."""
        feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)

        assert feat.has_value()
        assert not feat.has_path()
        assert not feat.is_placeholder()
        assert isinstance(feat.value, np.ndarray)
        assert feat.data_role == DataRole.FEATURE

    def test_value_entry_feature_with_torch_tensor(self, sample_torch_tensor: torch.Tensor):
        """Test ValueEntry (feature) with torch tensor."""
        feat = ValueEntry(name="test", value=sample_torch_tensor, data_role=DataRole.FEATURE)

        assert feat.has_value()
        assert isinstance(feat.value, torch.Tensor)

    def test_value_entry_feature_loss_input_default(self, sample_numpy_array: np.ndarray):
        """Test ValueEntry (feature) defaults loss_input to None."""
        feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)

        assert feat.loss_input is None


class TestValueEntryTarget:
    """Tests for ValueEntry with target role."""

    def test_value_entry_target_with_numpy_array(self, sample_numpy_array: np.ndarray):
        """Test ValueEntry (target) with numpy array."""
        targ = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.TARGET)

        assert targ.has_value()
        assert not targ.has_path()
        assert not targ.is_placeholder()
        assert isinstance(targ.value, np.ndarray)
        assert targ.data_role == DataRole.TARGET

    def test_value_entry_target_with_torch_tensor(self, sample_torch_tensor: torch.Tensor):
        """Test ValueEntry (target) with torch tensor."""
        targ = ValueEntry(name="test", value=sample_torch_tensor, data_role=DataRole.TARGET)

        assert targ.has_value()
        assert isinstance(targ.value, torch.Tensor)

    def test_value_entry_target_loss_input_default(self, sample_numpy_array: np.ndarray):
        """Test ValueEntry (target) defaults loss_input to None."""
        targ = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.TARGET)

        assert targ.loss_input is None

    def test_value_entry_target_write_attribute(self, sample_numpy_array: np.ndarray):
        """Test ValueEntry (target) has write attribute."""
        targ = ValueEntry(
            name="test", value=sample_numpy_array, data_role=DataRole.TARGET, write=True
        )

        assert targ.write is True


# ============================================================================
# Type Guard Tests
# ============================================================================


class TestTypeGuards:
    """Tests for type guard functions with new API."""

    def test_is_feature_npy(self, tmp_path: Path):
        """Test is_feature identifies NpyEntry with FEATURE role."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 2)))
        feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)
        targ = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.TARGET)

        assert is_feature(feat) is True
        assert is_feature(targ) is False

    def test_is_feature_value(self, sample_numpy_array: np.ndarray):
        """Test is_feature identifies ValueEntry with FEATURE role."""
        feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.TARGET)

        assert is_feature(feat) is True
        assert is_feature(targ) is False

    def test_is_target_npy(self, tmp_path: Path):
        """Test is_target identifies NpyEntry with TARGET role."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 1)))
        targ = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.TARGET)
        feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)

        assert is_target(targ) is True
        assert is_target(feat) is False

    def test_is_target_value(self, sample_numpy_array: np.ndarray):
        """Test is_target identifies ValueEntry with TARGET role."""
        targ = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.TARGET)
        feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)

        assert is_target(targ) is True
        assert is_target(feat) is False

    def test_is_path_based(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_path_based identifies path-based types."""
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 2)))
        path_feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)
        path_targ = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.TARGET)
        value_feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)

        assert is_path_based(path_feat) is True
        assert is_path_based(path_targ) is True
        assert is_path_based(value_feat) is False

    def test_is_value_based(self, tmp_path: Path, sample_numpy_array: np.ndarray):
        """Test is_value_based identifies value-based types."""
        value_feat = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.FEATURE)
        value_targ = ValueEntry(name="test", value=sample_numpy_array, data_role=DataRole.TARGET)
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 2)))
        path_feat = NpyEntry.model_construct(name="test", path=path, data_role=DataRole.FEATURE)

        assert is_value_based(value_feat) is True
        assert is_value_based(value_targ) is True
        assert is_value_based(path_feat) is False


class TestDatasetEntryFormatInference:
    @pytest.mark.parametrize(
        ("filename", "entry_type"),
        [
            ("x.npy", NpyEntry),
            ("x.npz", NpzEntry),
            ("x.csv", CsvEntry),
            ("x.txt", CsvEntry),
            ("x.parquet", ParquetEntry),
            ("x.h5", Hdf5Entry),
            ("x.hdf5", Hdf5Entry),
        ],
    )
    def test_path_entries_infer_format_from_suffix(
        self,
        tmp_path: Path,
        filename: str,
        entry_type: type[PathBasedEntry],
    ) -> None:
        path = tmp_path / filename
        path.write_bytes(b"placeholder")

        settings = DatasetSettings.model_validate(
            {"features": [{"name": "x", "path": path, "data_role": "feature"}]}
        )

        assert len(settings.features) == 1
        assert isinstance(settings.features[0], entry_type)

    def test_zarr_suffix_infers_zarr_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "store.zarr"
        path.mkdir()
        (path / "zarr.json").write_text("{}")

        settings = DatasetSettings.model_validate(
            {"features": [{"name": "x", "path": path, "data_role": "feature"}]}
        )

        assert len(settings.features) == 1
        assert isinstance(settings.features[0], ZarrEntry)

    def test_explicit_format_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "x.npy"
        path.write_bytes(b"placeholder")

        settings = DatasetSettings.model_validate(
            {"features": [{"name": "x", "path": path, "format": "npy", "data_role": "feature"}]}
        )

        assert isinstance(settings.features[0], NpyEntry)

    def test_placeholder_entry_without_path_is_preserved(self) -> None:
        settings = DatasetSettings.model_validate({"features": [{"name": "x"}]})

        assert len(settings.features) == 1
        assert isinstance(settings.features[0], ValueEntry)
        assert settings.features[0].is_placeholder()

    def test_value_entry_without_path_is_preserved(self, sample_numpy_array: np.ndarray) -> None:
        settings = DatasetSettings.model_validate(
            {"features": [{"name": "x", "value": sample_numpy_array}]}
        )

        assert len(settings.features) == 1
        assert isinstance(settings.features[0], ValueEntry)

    def test_autoencoder_target_without_format_is_preserved(self, tmp_path: Path) -> None:
        path = tmp_path / "target.npy"
        path.write_bytes(b"placeholder")

        settings = DatasetSettings.model_validate(
            {"targets": [{"name": "y", "path": path, "feature_ref": "x"}]}
        )

        assert len(settings.targets) == 1
        assert isinstance(settings.targets[0], AutoencoderTarget)

    def test_unknown_suffix_without_format_raises_clear_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "x.bin"
        path.write_bytes(b"placeholder")

        with pytest.raises(ValueError, match="Could not infer DATASET entry format"):
            DatasetSettings.model_validate({"features": [{"name": "x", "path": path}]})

    def test_plain_directory_without_zarr_suffix_raises_clear_validation_error(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "plain_dir"
        path.mkdir()

        with pytest.raises(ValueError, match="Directory-backed entries must use a '.zarr' suffix"):
            DatasetSettings.model_validate({"features": [{"name": "x", "path": path}]})

    def test_zarr_suffix_still_uses_existing_store_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.zarr"
        path.mkdir()

        with pytest.raises(ValueError, match="missing zarr.json"):
            DatasetSettings.model_validate({"features": [{"name": "x", "path": path}]})


# ============================================================================
# Strict Validation Mode Tests
# ============================================================================


class TestStrictValidation:
    """Tests for strict validation mode on NpyEntry."""

    def test_npy_entry_strict_mode_valid_path(self, tmp_path: Path):
        """Test NpyEntry passes strict validation with existing path."""
        test_file = tmp_path / "test.npy"
        np.save(test_file, np.ones((10, 5)))

        feat = NpyEntry.model_validate(
            {"name": "test", "path": str(test_file), "data_role": "feature"},
            context={"strict": True},
        )

        assert feat.has_path()

    def test_npy_entry_strict_mode_invalid_path(self):
        """Test NpyEntry fails strict validation with non-existing path."""
        with pytest.raises(ValueError, match="does not exist"):
            NpyEntry.model_validate(
                {"name": "test", "path": "/nonexistent/path.npy", "data_role": "feature"},
                context={"strict": True},
            )

    def test_npy_entry_non_strict_mode_invalid_path(self):
        """Test NpyEntry constructed with non-strict mode allows non-existing path."""
        # Use model_construct to create without validation
        feat = NpyEntry.model_construct(
            name="test", path="/nonexistent/path.npy", data_role=DataRole.FEATURE
        )

        assert feat.has_path()

    def test_npy_entry_placeholder_strict_mode_ok(self):
        """Test placeholder passes strict validation (no path to check)."""
        feat = NpyEntry.model_validate(
            {"name": "test", "data_role": "feature"}, context={"strict": True}
        )

        assert feat.is_placeholder()


# ============================================================================
# DatasetSettings Integration Tests
# ============================================================================


class TestDatasetSettingsValuePreservation:
    """Regression tests for ValueEntry preservation in DatasetSettings.

    These tests guard against the bug where nested_model_default_partial_update=True
    caused ValueEntry objects to be converted to NpyEntry objects during
    DatasetSettings validation (losing the in-memory value field).
    """

    def test_value_entry_preserved_in_dataset_settings(self, sample_numpy_array: np.ndarray):
        """ValueEntry should remain ValueEntry when stored in DatasetSettings.features."""
        feat = ValueEntry(name="x", value=sample_numpy_array, data_role=DataRole.FEATURE)
        assert isinstance(feat, ValueEntry)
        assert feat.has_value()
        feat_id = id(feat)

        # Create DatasetSettings with ValueEntry
        ds_settings = DatasetSettings(features=(feat,))

        # Verify type preservation
        assert isinstance(ds_settings.features[0], ValueEntry)

        # Verify value preservation
        assert ds_settings.features[0].has_value()
        assert ds_settings.features[0].value is not None
        assert ds_settings.features[0].value.shape == sample_numpy_array.shape

        # Verify object identity (no deep copy)
        assert id(ds_settings.features[0]) == feat_id
        assert ds_settings.features[0] is feat

    def test_value_entry_target_preserved_in_dataset_settings(self, sample_numpy_array: np.ndarray):
        """ValueEntry (target) should remain ValueEntry when stored in DatasetSettings.targets."""
        targ = ValueEntry(name="y", value=sample_numpy_array, data_role=DataRole.TARGET)
        assert isinstance(targ, ValueEntry)
        assert targ.has_value()
        targ_id = id(targ)

        # Create DatasetSettings with ValueEntry target
        ds_settings = DatasetSettings(features=(), targets=(targ,))

        # Verify type preservation
        assert isinstance(ds_settings.targets[0], ValueEntry)

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
        """DatasetSettings should handle mixed ValueEntry and NpyEntry correctly."""
        # Create path-based feature
        x_path = tmp_path / "x.npy"
        np.save(x_path, sample_numpy_array)
        path_feat = NpyEntry(name="x_path", path=x_path, data_role=DataRole.FEATURE)

        # Create value-based feature
        value_feat = ValueEntry(
            name="x_value", value=sample_numpy_array, data_role=DataRole.FEATURE
        )

        # Create DatasetSettings with mixed entries
        ds_settings = DatasetSettings(features=(path_feat, value_feat))

        # Verify path feature is NpyEntry (PathBasedEntry)
        assert isinstance(ds_settings.features[0], NpyEntry)
        assert isinstance(ds_settings.features[0], PathBasedEntry)
        assert ds_settings.features[0].has_path()
        assert not ds_settings.features[0].has_value()

        # Verify value feature is ValueEntry (ValueBasedEntry)
        assert isinstance(ds_settings.features[1], ValueEntry)
        assert isinstance(ds_settings.features[1], ValueBasedEntry)
        assert ds_settings.features[1].has_value()
        assert not ds_settings.features[1].has_path()

        # Verify object identity
        assert ds_settings.features[0] is path_feat
        assert ds_settings.features[1] is value_feat

    def test_value_entries_work_with_flexible_dataset(self, sample_numpy_array: np.ndarray):
        """ValueEntry should work with FlexibleDataset when passed through DatasetSettings."""
        from dlkit.engine.data.datasets.flexible import FlexibleDataset

        # Create value-based entries
        feat = ValueEntry(name="x", value=sample_numpy_array, data_role=DataRole.FEATURE)
        targ = ValueEntry(name="y", value=sample_numpy_array[:, :1], data_role=DataRole.TARGET)

        # Create DatasetSettings
        ds_settings = DatasetSettings(features=(feat,), targets=(targ,))

        # Create FlexibleDataset using features/targets from DatasetSettings
        dataset = FlexibleDataset(entries=list(ds_settings.features) + list(ds_settings.targets))

        # Verify dataset works
        assert len(dataset) == 10
        sample = dataset[0]
        features = cast(object, sample["features"])
        targets = cast(object, sample["targets"])
        assert len(list(cast("dict[str, torch.Tensor]", features).keys())) == 1
        assert len(list(cast("dict[str, torch.Tensor]", targets).keys())) == 1
        assert sample["features", "x"].shape == torch.Size([5])
        assert sample["targets", "y"].shape == torch.Size([1])
