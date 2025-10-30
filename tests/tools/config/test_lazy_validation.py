"""Tests for lazy validation architecture.

This module tests the new lazy validation system where:
1. Configs load without validation (validate=False, default)
2. Validation happens at build time via BuildFactory
3. Programmatic overrides work with incomplete configs
"""

import pytest
from pathlib import Path
from pydantic import ValidationError
import numpy as np

from dlkit.tools.io.config import load_sections_config
from dlkit.tools.config import load_training_settings, update_settings
from dlkit.tools.config.dataset_settings import DatasetSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.runtime.workflows.factories.build_factory import BuildFactory


class TestLazyValidationConfigLoading:
    """Test lazy loading of configs without validation."""

    def test_load_config_with_missing_dataset_paths_lazy_mode(self, tmp_path):
        """Test that configs with placeholder paths load successfully in lazy mode."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[DATASET]
name = "FlexibleDataset"
module_path = "dlkit.core.datasets"

[[DATASET.features]]
name = "x"
path = "/nonexistent/placeholder.npy"

[[DATASET.targets]]
name = "y"
path = "/another/nonexistent/path.npy"
""")

        # Load with lazy validation (default) - should NOT fail
        result = load_sections_config(config_path, {"DATASET": DatasetSettings}, validate=False)
        dataset_settings = result["DATASET"]

        # Verify loaded successfully despite invalid paths
        assert dataset_settings is not None
        # In lazy mode with model_construct fallback, nested objects may be dicts or models
        # Just verify they loaded
        features = dataset_settings.features
        targets = dataset_settings.targets
        assert len(features) == 1
        assert len(targets) == 1
        # Access name safely (could be dict or Feature object)
        feat_name = features[0].name if hasattr(features[0], 'name') else features[0]['name']
        targ_name = targets[0].name if hasattr(targets[0], 'name') else targets[0]['name']
        assert feat_name == "x"
        assert targ_name == "y"

    def test_load_config_with_missing_paths_strict_mode_fails(self, tmp_path):
        """Test that strict validation (validate=True) fails with invalid paths."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[DATASET]
name = "FlexibleDataset"
module_path = "dlkit.core.datasets"

[[DATASET.features]]
name = "x"
path = "/nonexistent/placeholder.npy"
""")

        # Load with strict validation - SHOULD fail
        with pytest.raises(Exception):  # Could be ConfigValidationError or ValidationError
            load_sections_config(config_path, {"DATASET": DatasetSettings}, validate=True)

    def test_load_training_settings_with_incomplete_dataset(self, tmp_path):
        """Test that training settings load with completely empty DATASET section."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test_session"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Empty dataset section - no features or targets!

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Should load successfully despite incomplete DATASET
        settings = load_training_settings(config_path)

        assert settings is not None
        assert settings.SESSION is not None
        assert settings.SESSION.name == "test_session"
        assert settings.DATASET is not None
        # Features and targets are empty
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0


class TestProgrammaticOverrides:
    """Test programmatic path injection via overrides."""

    def test_override_dataset_paths_with_model_copy(self, tmp_path):
        """Test that dataset paths can be injected via update_settings after lazy load."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Create config with incomplete DATASET
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_override"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Empty initially

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load config with lazy validation
        settings = load_training_settings(config_path)
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0

        # Apply overrides using clean update_settings helper
        settings = update_settings(settings, {
            "DATASET": {
                "features": (Feature(name="x", path=features_path),),
                "targets": (Target(name="y", path=targets_path),),
            }
        })

        # Verify overrides applied
        assert len(settings.DATASET.features) == 1
        assert len(settings.DATASET.targets) == 1
        assert Path(settings.DATASET.features[0].path).exists()
        assert Path(settings.DATASET.targets[0].path).exists()


class TestPreBuildValidation:
    """Test that validation happens at build time in BuildFactory."""

    def test_build_factory_catches_invalid_paths_at_validation_time(self, tmp_path):
        """Test that BuildFactory._validate_settings() catches invalid paths when programmatically modified."""
        # Create real data files first
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Create config with valid paths
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_bad_paths"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "{features_path.as_posix()}"

[[DATASET.targets]]
name = "y"
path = "{targets_path.as_posix()}"

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load config (succeeds)
        settings = load_training_settings(config_path)
        assert settings is not None

        # Now programmatically modify to have invalid paths using model_construct to bypass validation
        from dlkit.tools.config.data_entries import Feature, Target

        # Use model_construct to create features with bad paths (bypasses validation)
        bad_features = (
            Feature.model_construct(name="x", path="/this/path/does/not/exist.npy"),
        )
        bad_targets = (
            Target.model_construct(name="y", path="/another/bad/path.npy"),
        )

        # Modify dataset with bad paths
        bad_dataset = settings.DATASET.model_construct(
            name="FlexibleDataset",
            features=bad_features,
            targets=bad_targets,
        )
        settings = settings.model_construct(**{
            **settings.model_dump(),
            "DATASET": bad_dataset
        })

        # Try to build - should fail at validation time
        factory = BuildFactory()
        with pytest.raises(ValidationError) as exc_info:
            factory.build_components(settings)

        # Verify the error mentions the path validation failure
        error_message = str(exc_info.value)
        assert "path" in error_message.lower() or "file" in error_message.lower()

    def test_build_factory_fails_fast_with_incomplete_settings(self, tmp_path):
        """Test that build fails fast when settings are incomplete (even if validation passes)."""
        # Create real data files so paths are valid
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_fail_fast"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "{features_path.as_posix()}"

[[DATASET.targets]]
name = "y"
path = "{targets_path.as_posix()}"

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load and verify settings are valid
        settings = load_training_settings(config_path)
        assert settings is not None

        # Now corrupt the settings by removing TRAINING using model_construct
        # (model_copy won't work because TRAINING is optional now)
        corrupted_settings = settings.__class__.model_construct(
            **{**settings.model_dump(), "TRAINING": None}
        )

        # Try to build - should fail fast (AttributeError when accessing TRAINING.optimizer)
        factory = BuildFactory()
        with pytest.raises((ValidationError, AttributeError)):
            # Either validation catches it or build process fails fast
            factory.build_components(corrupted_settings)

    def test_settings_can_be_loaded_and_modified_programmatically(self, tmp_path):
        """Test that settings can be loaded with lazy validation and modified programmatically."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Create config
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_settings"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Initially empty

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load config (succeeds with lazy validation)
        settings = load_training_settings(config_path)
        assert settings is not None
        assert len(settings.DATASET.features) == 0

        # Modify programmatically
        new_dataset = settings.DATASET.model_copy(update={
            "features": (Feature(name="x", path=features_path),),
            "targets": (Target(name="y", path=targets_path),),
        })
        settings = settings.model_copy(update={"DATASET": new_dataset})

        # Verify modifications
        assert len(settings.DATASET.features) == 1
        assert len(settings.DATASET.targets) == 1
        assert Path(settings.DATASET.features[0].path).exists()
        assert Path(settings.DATASET.targets[0].path).exists()

    def test_settings_structure_is_valid(self, tmp_path):
        """Test that settings loaded with lazy validation have correct structure."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Create config with valid paths
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_structure"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "{features_path.as_posix()}"

[[DATASET.targets]]
name = "y"
path = "{targets_path.as_posix()}"

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load settings
        settings = load_training_settings(config_path)

        # Verify structure
        assert settings is not None
        assert settings.SESSION is not None
        assert settings.DATAMODULE is not None
        assert settings.DATASET is not None
        assert settings.MODEL is not None
        assert settings.TRAINING is not None
        assert len(settings.DATASET.features) == 1
        assert len(settings.DATASET.targets) == 1


class TestEndToEndLazyValidation:
    """End-to-end test of lazy validation workflow."""

    def test_complete_lazy_validation_workflow(self, tmp_path):
        """Test complete workflow: lazy load → override → settings ready."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        # Create config with incomplete DATASET
        config_path = tmp_path / "config.toml"
        config_path.write_text(f"""
[SESSION]
name = "test_e2e"
root_dir = "{tmp_path.as_posix()}"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Initially empty - will be overridden programmatically

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Step 1: Load with lazy validation (succeeds despite empty DATASET)
        settings = load_training_settings(config_path)
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0

        # Step 2: Apply programmatic overrides
        new_dataset = settings.DATASET.model_copy(update={
            "features": (Feature(name="x", path=features_path),),
            "targets": (Target(name="y", path=targets_path),),
        })
        settings = settings.model_copy(update={"DATASET": new_dataset})

        # Step 3: Verify settings are complete and ready
        assert len(settings.DATASET.features) == 1
        assert len(settings.DATASET.targets) == 1
        assert Path(settings.DATASET.features[0].path).exists()
        assert Path(settings.DATASET.targets[0].path).exists()

        # Settings are now complete and could be used for building components
        assert settings.SESSION is not None
        assert settings.DATAMODULE is not None
        assert settings.MODEL is not None
        assert settings.TRAINING is not None


class TestPartialDatasetValidation:
    """Test partial dataset configuration with some fields present but not others."""

    def test_dataset_with_split_settings_but_no_features_targets(self, tmp_path):
        """Test that DATASET can have split settings (val_ratio) without features/targets.

        Note: Split settings must be in [DATASET.split] subsection, not directly in [DATASET].
        """
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test_partial"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Empty main section

[DATASET.split]
# Has split settings but NO features or targets in parent section
val_ratio = 0.2
test_ratio = 0.1

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load with lazy validation - should succeed
        settings = load_training_settings(config_path)

        # Verify split settings loaded correctly
        assert settings.DATASET is not None
        assert settings.DATASET.split.val_ratio == 0.2
        assert settings.DATASET.split.test_ratio == 0.1
        assert settings.DATASET.split.train_ratio == 0.7  # 1.0 - 0.2 - 0.1

        # Features and targets should be empty
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0

    def test_dataset_with_nested_split_section_but_no_data(self, tmp_path):
        """Test that DATASET can have [DATASET.split] section without features/targets."""
        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test_partial_nested"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Empty main section

[DATASET.split]
val_ratio = 0.25
test_ratio = 0.15

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load with lazy validation - should succeed
        settings = load_training_settings(config_path)

        # Verify split settings loaded correctly
        assert settings.DATASET is not None
        assert settings.DATASET.split.val_ratio == 0.25
        assert settings.DATASET.split.test_ratio == 0.15
        assert settings.DATASET.split.train_ratio == 0.6  # 1.0 - 0.25 - 0.15

        # Features and targets should be empty
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0

    def test_partial_dataset_can_be_completed_programmatically(self, tmp_path):
        """Test that partial DATASET with splits can be completed with data paths."""
        # Create real data files
        features_path = tmp_path / "features.npy"
        targets_path = tmp_path / "targets.npy"
        np.save(features_path, np.random.rand(100, 10))
        np.save(targets_path, np.random.rand(100, 1))

        config_path = tmp_path / "config.toml"
        config_path.write_text("""
[SESSION]
name = "test_partial_complete"

[DATAMODULE]
name = "InMemoryModule"
module_path = "dlkit.core.datamodules"
batch_size = 16

[DATASET]
# Empty main section

[DATASET.split]
# Has split settings but no data paths - will add them programmatically
val_ratio = 0.2

[MODEL]
name = "LinearNetwork"
module_path = "dlkit.core.models.nn.ffnn"
input_size = 10
output_size = 1

[TRAINING]
epochs = 1

[TRAINING.optimizer]
name = "Adam"
lr = 0.001

[TRAINING.loss_function]
name = "MSELoss"
module_path = "torch.nn"
""")

        # Load with lazy validation
        settings = load_training_settings(config_path)

        # Verify split settings preserved
        assert settings.DATASET.split.val_ratio == 0.2
        assert len(settings.DATASET.features) == 0
        assert len(settings.DATASET.targets) == 0

        # Now add features and targets programmatically while preserving split settings
        new_dataset = settings.DATASET.model_copy(update={
            "features": (Feature(name="x", path=features_path),),
            "targets": (Target(name="y", path=targets_path),),
        })
        settings = settings.model_copy(update={"DATASET": new_dataset})

        # Verify both split settings and data paths are present
        assert settings.DATASET.split.val_ratio == 0.2
        assert len(settings.DATASET.features) == 1
        assert len(settings.DATASET.targets) == 1
        assert Path(settings.DATASET.features[0].path).exists()
        assert Path(settings.DATASET.targets[0].path).exists()
