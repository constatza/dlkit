"""Tests for settings updater with deep merge functionality."""

import pytest
from pathlib import Path
from pydantic import ValidationError
import numpy as np

from dlkit.tools.config import update_settings, load_settings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.training_settings import TrainingSettings
from dlkit.tools.config.optimizer_settings import OptimizerSettings


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

        # Update SESSION.name
        new_settings = update_settings(settings, {
            "SESSION": {"name": "new_name"}
        })

        # Name should be updated
        assert new_settings.SESSION.name == "new_name"

        # Other SESSION fields should be preserved
        assert new_settings.SESSION.seed == settings.SESSION.seed
        assert new_settings.SESSION.precision == settings.SESSION.precision

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

        # Update TRAINING.trainer.max_epochs
        new_settings = update_settings(settings, {
            "TRAINING": {
                "trainer": {
                    "max_epochs": 200
                }
            }
        })

        # max_epochs should be updated
        assert new_settings.TRAINING.trainer.max_epochs == 200

        # Other trainer fields should be preserved
        assert new_settings.TRAINING.trainer.name == settings.TRAINING.trainer.name

        # Other TRAINING fields should be preserved
        assert new_settings.TRAINING.epochs == settings.TRAINING.epochs
        assert new_settings.TRAINING.optimizer.lr == settings.TRAINING.optimizer.lr

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

        # Update multiple sections
        new_settings = update_settings(settings, {
            "SESSION": {"name": "updated_session"},
            "TRAINING": {"epochs": 100},
            "DATAMODULE": {"dataloader": {"batch_size": 64}}
        })

        # All updates should be applied
        assert new_settings.SESSION.name == "updated_session"
        assert new_settings.TRAINING.epochs == 100
        assert new_settings.DATAMODULE.dataloader.batch_size == 64

        # Unspecified fields preserved
        assert new_settings.TRAINING.optimizer.lr == settings.TRAINING.optimizer.lr


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

        # Update only lr
        new_settings = update_settings(settings, {
            "TRAINING": {
                "optimizer": {
                    "lr": 0.01
                }
            }
        })

        # lr should be updated
        assert new_settings.TRAINING.optimizer.lr == 0.01

        # weight_decay should be preserved
        assert new_settings.TRAINING.optimizer.weight_decay == 0.01

        # optimizer.name should be preserved
        assert new_settings.TRAINING.optimizer.name == "Adam"


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
        new_settings = update_settings(settings, {
            "DATASET": {
                "features": (
                    Feature(name="new_feature1", path=features_path),
                    Feature(name="new_feature2", path=features_path),
                )
            }
        })

        # Features list should be completely replaced
        assert len(new_settings.DATASET.features) == 2
        assert new_settings.DATASET.features[0].name == "new_feature1"
        assert new_settings.DATASET.features[1].name == "new_feature2"

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

        new_settings = update_settings(settings, {
            "SESSION": {"name": "updated"}
        })

        assert new_settings.SESSION.name == "updated"

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

        new_settings = update_settings(settings, {
            "TRAINING": {"epochs": 999}
        })

        assert new_settings.TRAINING.epochs == 999

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

        new_settings = update_settings(settings, {
            "MODEL": {"checkpoint": ckpt2}
        })

        assert new_settings.MODEL.checkpoint == ckpt2




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
        assert settings.TRAINING.optimizer.name == "Adam"
        assert settings.TRAINING.optimizer.lr == 0.005
        assert settings.TRAINING.optimizer.weight_decay == 0.1

        updated = update_settings(settings, {"TRAINING": TrainingSettings(epochs=50)})

        assert updated.TRAINING.optimizer.name == "Adam"
        assert updated.TRAINING.optimizer.lr == pytest.approx(0.005)
        assert updated.TRAINING.optimizer.weight_decay == pytest.approx(0.1)

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
        assert settings.TRAINING.optimizer.name == "SGD"
        assert settings.TRAINING.optimizer.weight_decay == 0.2

        updated = update_settings(
            settings,
            {"TRAINING": {"optimizer": OptimizerSettings(lr=0.123)}},
        )

        assert updated.TRAINING.optimizer.name == "SGD"
        assert updated.TRAINING.optimizer.weight_decay == pytest.approx(0.2)
        assert updated.TRAINING.optimizer.lr == pytest.approx(0.123)

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
        new_settings = update_settings(settings, {
            "EXTRAS": {
                "deeply": {
                    "nested": {
                        "custom": "data"
                    }
                }
            }
        })

        # Original keys remain intact
        assert getattr(new_settings.EXTRAS, "custom_field1") == "value1"
        assert getattr(new_settings.EXTRAS, "custom_field2") == "value2"

        # New nested content is merged in
        assert hasattr(new_settings.EXTRAS, "deeply")
        assert new_settings.EXTRAS.deeply["nested"]["custom"] == "data"

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

        new_settings = update_settings(settings, {
            "EXTRAS": {
                "custom_field2": "updated",
            }
        })

        assert getattr(new_settings.EXTRAS, "custom_field1") == "value1"
        assert getattr(new_settings.EXTRAS, "custom_field2") == "updated"


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
            update_settings(settings, {
                "DATASET": {
                    "features": (
                        Feature(name="x", path="/nonexistent/bad/path.npy"),
                    )
                }
            })

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
        new_settings = update_settings(settings, {
            "SESSION": {"name": "updated"}
        })

        # Should complete successfully with validation
        assert new_settings.SESSION.name == "updated"


class TestMutability:
    """Test that settings are mutated in-place (no longer immutable)."""

    def test_original_settings_mutated(self, tmp_path):
        """Test that update mutates the same instance (not a copy)."""
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

        # Capture original id
        original_id = id(settings)

        # Update settings
        new_settings = update_settings(settings, {
            "SESSION": {"name": "modified"},
            "TRAINING": {"epochs": 999}
        })

        # Should be the same instance (mutated in-place)
        assert id(new_settings) == original_id
        assert new_settings is settings

        # Original IS modified (mutation, not copy)
        assert settings.SESSION.name == "modified"
        assert settings.TRAINING.epochs == 999


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
        new_settings = update_settings(settings, {
            "EXTRAS": {
                "existing_field": "value",  # Keep existing
                "new_field": "new_value"     # Add new
            }
        })

        # EXTRAS is ExtrasSettings with extra="allow", access via attributes
        assert new_settings.EXTRAS.existing_field == "value"
        assert new_settings.EXTRAS.new_field == "new_value"

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

        assert via_method.SESSION.name == "updated_name"
        # update_with returns the same instance (mutation)
        assert via_method is settings
        # Original instance IS changed (mutation, not immutability)
        assert settings.SESSION.name == "updated_name"
