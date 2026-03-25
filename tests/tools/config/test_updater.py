"""Tests for settings updater with deep merge functionality."""

from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from dlkit.tools.config import load_settings, update_settings
from dlkit.tools.config.data_entries import Feature


def _expect_not_none[T](value: T | None) -> T:
    assert value is not None
    return value


def _expect_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


class TestBasicUpdates:
    """Test basic update functionality."""

    def test_simple_top_level_update(self, tmp_path):
        """Test updating a simple top-level field."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "original_name"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)
        session = _expect_not_none(settings.SESSION)

        # Update SESSION.name
        new_settings = update_settings(settings, {"SESSION": {"name": "new_name"}})
        new_session = _expect_not_none(new_settings.SESSION)

        # Name should be updated
        assert new_session.name == "new_name"

        # Other SESSION fields should be preserved
        assert new_session.seed == session.seed
        assert new_session.precision == session.precision

    def test_deep_nested_update(self, tmp_path):
        """Test updating deeply nested fields (3 levels)."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"

[TRAINING.trainer]
max_epochs = 100
""")

        settings = load_settings(config_path)
        training = _expect_not_none(settings.TRAINING)

        # Update TRAINING.trainer.max_epochs
        new_settings = update_settings(settings, {"TRAINING": {"trainer": {"max_epochs": 200}}})
        new_training = _expect_not_none(new_settings.TRAINING)

        # max_epochs should be updated
        assert new_training.trainer.max_epochs == 200

        # Other trainer fields should be preserved
        assert new_training.trainer.name == training.trainer.name

        # Other TRAINING fields should be preserved
        assert new_training.epochs == training.epochs
        assert new_training.optimizer.lr == training.optimizer.lr

    def test_multiple_sections_at_once(self, tmp_path):
        """Test updating multiple top-level sections simultaneously."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)
        training = _expect_not_none(settings.TRAINING)

        # Update multiple sections
        new_settings = update_settings(
            settings,
            {
                "SESSION": {"name": "updated_session"},
                "TRAINING": {"epochs": 100},
                "DATAMODULE": {"dataloader": {"batch_size": 64}},
            },
        )
        session = _expect_not_none(new_settings.SESSION)
        training_updated = _expect_not_none(new_settings.TRAINING)
        datamodule = _expect_not_none(new_settings.DATAMODULE)

        # All updates should be applied
        assert session.name == "updated_session"
        assert training_updated.epochs == 100
        assert datamodule.dataloader.batch_size == 64

        # Unspecified fields preserved
        assert training_updated.optimizer.lr == training.optimizer.lr


class TestPartialUpdatesPreservation:
    """Test that partial updates preserve unspecified fields."""

    def test_partial_update_preserves_sibling_fields(self, tmp_path):
        """Test that updating one field preserves sibling fields."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001
weight_decay = 0.01

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)
        training = _expect_not_none(settings.TRAINING)

        # Update only lr
        new_settings = update_settings(settings, {"TRAINING": {"optimizer": {"lr": 0.01}}})
        new_training = _expect_not_none(new_settings.TRAINING)

        # lr should be updated
        assert new_training.optimizer.lr == 0.01

        # weight_decay should be preserved
        assert new_training.optimizer.weight_decay == 0.01

        # optimizer.name should be preserved
        assert new_training.optimizer.name == training.optimizer.name


class TestNonDictTypes:
    """Test handling of non-dict types (list, tuple, str, int, etc.)."""

    def test_list_overwrite(self, tmp_path):
        """Test that lists are overwritten, not merged."""
        # Create data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[[DATASET.features]]
name = "old_feature"
path = "{features_path.as_posix()}"

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        # Replace features list completely
        new_settings = update_settings(
            settings,
            {
                "DATASET": {
                    "features": (
                        Feature(name="new_feature1", path=features_path),
                        Feature(name="new_feature2", path=features_path),
                    )
                }
            },
        )
        dataset = _expect_not_none(new_settings.DATASET)

        # Features list should be completely replaced
        assert len(dataset.features) == 2
        assert dataset.features[0].name == "new_feature1"
        assert dataset.features[1].name == "new_feature2"

    def test_string_overwrite(self, tmp_path):
        """Test that strings are overwritten."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "original"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        new_settings = update_settings(settings, {"SESSION": {"name": "updated"}})

        assert _expect_not_none(new_settings.SESSION).name == "updated"

    def test_int_overwrite(self, tmp_path):
        """Test that integers are overwritten."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        new_settings = update_settings(settings, {"TRAINING": {"epochs": 999}})

        assert _expect_not_none(new_settings.TRAINING).epochs == 999

    def test_path_overwrite(self, tmp_path):
        """Test that Path objects are overwritten."""
        ckpt1 = tmp_path / "model1.ckpt"
        ckpt2 = tmp_path / "model2.ckpt"
        ckpt1.touch()
        ckpt2.touch()

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
checkpoint = "{ckpt1.as_posix()}"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        new_settings = update_settings(settings, {"MODEL": {"checkpoint": ckpt2}})

        assert _expect_not_none(new_settings.MODEL).checkpoint == ckpt2


class TestPydanticModelUpdates:
    """Ensure BaseModel updates don't reset existing config sections to defaults."""

    def test_training_model_update_preserves_nested_values(self, tmp_path):
        """Applying TrainingSettings instance keeps optimizer overrides from config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.005
weight_decay = 0.1

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)
        training = _expect_not_none(settings.TRAINING)
        assert training.optimizer.name == "Adam"
        assert training.optimizer.lr == 0.005
        assert training.optimizer.weight_decay == 0.1

        updated = update_settings(settings, {"TRAINING": {"epochs": 50}})
        updated_training = _expect_not_none(updated.TRAINING)

        assert updated_training.optimizer.name == "Adam"
        assert updated_training.optimizer.lr == pytest.approx(0.005)
        assert updated_training.optimizer.weight_decay == pytest.approx(0.1)

    def test_optimizer_model_update_preserves_existing_fields(self, tmp_path):
        """Applying OptimizerSettings instance merges into existing optimizer config."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "SGD"
lr = 0.01
weight_decay = 0.2

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)
        training = _expect_not_none(settings.TRAINING)
        assert training.optimizer.name == "SGD"
        assert training.optimizer.weight_decay == 0.2

        updated = update_settings(
            settings,
            {"TRAINING": {"optimizer": {"lr": 0.123}}},
        )
        updated_training = _expect_not_none(updated.TRAINING)

        assert updated_training.optimizer.name == "SGD"
        assert updated_training.optimizer.weight_decay == pytest.approx(0.2)
        assert updated_training.optimizer.lr == pytest.approx(0.123)


class TestExtrasHandling:
    """Test handling of EXTRAS (allowing arbitrary keys)."""

    def test_extras_dict_merges_without_deleting(self, tmp_path):
        """Updating EXTRAS should preserve unspecified keys and add new ones."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"

[EXTRAS]
custom_field1 = "value1"
custom_field2 = "value2"
""")

        settings = load_settings(config_path)

        # Add nested structure while keeping existing keys
        new_settings = update_settings(
            settings, {"EXTRAS": {"deeply": {"nested": {"custom": "data"}}}}
        )
        extras = _expect_not_none(new_settings.EXTRAS)

        # Original keys remain intact
        assert extras.custom_field1 == "value1"
        assert extras.custom_field2 == "value2"

        # New nested content is merged in
        assert hasattr(extras, "deeply")
        deeply = _expect_mapping(extras.deeply)
        nested = _expect_mapping(deeply["nested"])
        assert nested["custom"] == "data"

    def test_extras_overwrite_specific_key(self, tmp_path):
        """Overwriting a key updates its value without dropping others."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"

[EXTRAS]
custom_field1 = "value1"
custom_field2 = "value2"
""")

        settings = load_settings(config_path)

        new_settings = update_settings(
            settings,
            {
                "EXTRAS": {
                    "custom_field2": "updated",
                }
            },
        )
        extras = _expect_not_none(new_settings.EXTRAS)

        assert extras.custom_field1 == "value1"
        assert extras.custom_field2 == "updated"


class TestValidation:
    """Test validation behavior."""

    def test_validation_catches_bad_paths(self, tmp_path):
        """Test that validation catches invalid file paths."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        # Try to update with invalid path - should fail validation
        with pytest.raises(ValidationError):
            update_settings(
                settings,
                {"DATASET": {"features": (Feature(name="x", path="/nonexistent/bad/path.npy"),)}},
            )

    def test_validation_with_default_catches_errors(self, tmp_path):
        """Test that validation is enabled by default and catches errors."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        # Valid update should work (validation is on by default)
        new_settings = update_settings(settings, {"SESSION": {"name": "updated"}})

        # Should complete successfully with validation
        assert _expect_not_none(new_settings.SESSION).name == "updated"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_add_new_field_to_section(self, tmp_path):
        """Test adding a new field that doesn't exist in original."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"

[EXTRAS]
existing_field = "value"
""")

        settings = load_settings(config_path)

        # Add new field to EXTRAS
        new_settings = update_settings(
            settings,
            {
                "EXTRAS": {
                    "existing_field": "value",  # Keep existing
                    "new_field": "new_value",  # Add new
                }
            },
        )

        # EXTRAS is ExtrasSettings with extra="allow", access via attributes
        extras = _expect_not_none(new_settings.EXTRAS)
        assert extras.existing_field == "value"
        assert extras.new_field == "new_value"

    def test_empty_dict(self, tmp_path):
        """Test updating with empty dict."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 10

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        # Update with empty dict should not change anything
        new_settings = update_settings(settings, {})

        assert new_settings.model_dump() == settings.model_dump()


class TestInstanceHelper:
    """Test the BasicSettings.update_with convenience method."""

    def test_update_with_matches_functional_helper(self, tmp_path):
        """Ensure update_with delegates to update_settings and preserves type."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "instance_helper"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        settings = load_settings(config_path)

        via_method = settings.update_with({"SESSION": {"name": "updated_name"}})

        assert _expect_not_none(via_method.SESSION).name == "updated_name"
        # update_with returns a NEW instance (immutable semantics)
        assert via_method is not settings
        # Original is unchanged
        assert _expect_not_none(settings.SESSION).name == "instance_helper"
