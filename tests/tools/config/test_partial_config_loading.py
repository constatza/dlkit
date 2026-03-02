"""Tests for partial config loading to verify training vs inference sections."""

from pathlib import Path

import pytest

from dlkit.tools.config import (
    load_settings,
    load_sections,
)
from dlkit.tools.config.workflow_settings import (
    TrainingWorkflowSettings,
    InferenceWorkflowSettings,
)


@pytest.fixture
def full_config_content():
    """Sample config with all sections for testing."""
    return """
[SESSION]
name = "test_session"
inference = false

[MODEL]
name = "TestModel"
module_path = "test.model"

[DATAMODULE]
name = "TestDataModule"

[DATAMODULE.dataloader]
batch_size = 32

[DATASET]
name = "TestDataset"
type = "flexible"

[TRAINING]
epochs = 10

[TRAINING.trainer]
accelerator = "cpu"

[MLFLOW]
enabled = false

[OPTUNA]
enabled = false

[PATHS]
output_dir = "./output"

[EXTRAS]
custom_param = "test"
"""


@pytest.fixture
def config_file(tmp_path: Path, full_config_content: str):
    """Create temporary config file for testing."""
    config_path = tmp_path / "full_config.toml"
    config_path.write_text(full_config_content)
    return config_path


class TestPartialConfigLoading:
    """Test partial config loading for training vs inference."""

    def test_training_settings_loads_all_sections(self, config_file):
        """Training settings should load all sections including TRAINING, MLFLOW, OPTUNA."""
        settings = load_settings(config_file)

        # Verify correct type
        assert isinstance(settings, TrainingWorkflowSettings)

        # Verify all sections are loaded
        assert hasattr(settings, "SESSION")
        assert hasattr(settings, "MODEL")
        assert hasattr(settings, "DATAMODULE")
        assert hasattr(settings, "DATASET")
        assert hasattr(settings, "TRAINING")
        assert hasattr(settings, "MLFLOW")
        assert hasattr(settings, "OPTUNA")
        assert hasattr(settings, "PATHS")
        assert hasattr(settings, "EXTRAS")

        # Verify sections are not None
        assert settings.SESSION is not None
        assert settings.DATAMODULE is not None
        assert settings.DATASET is not None
        assert settings.TRAINING is not None

        # Verify training-specific properties work
        assert settings.has_training_config is True
        assert settings.mlflow_enabled is False

    # REMOVED: Inference workflow removed in breaking change
    # def test_inference_settings_excludes_training_sections(self, config_file):
    #     """Inference settings should exclude TRAINING, MLFLOW, OPTUNA sections."""
    #     settings = load_inference_settings(config_file)

    #     # Verify correct type
    #     assert isinstance(settings, InferenceWorkflowSettings)
    #
    #     # Verify core sections are loaded
    #     assert hasattr(settings, 'SESSION')
    #     assert hasattr(settings, 'MODEL')
    #     assert hasattr(settings, 'DATAMODULE')
    #     assert hasattr(settings, 'DATASET')
    #     assert hasattr(settings, 'PATHS')
    #     assert hasattr(settings, 'EXTRAS')
    #
    #     # Verify training sections are NOT loaded
    #     assert not hasattr(settings, 'TRAINING')
    #     assert not hasattr(settings, 'MLFLOW')
    #     assert not hasattr(settings, 'OPTUNA')
    #
    #     # Verify sections are not None
    #     assert settings.SESSION is not None
    #     assert settings.DATAMODULE is not None
    #     assert settings.DATASET is not None

    # def test_inference_settings_has_correct_interface(self, config_file):
    #     """Inference settings should implement the inference protocol."""
    #     settings = load_inference_settings(config_file)

    #     # Verify inference-specific properties
    #     assert hasattr(settings, 'checkpoint_path')
    #     # checkpoint_path can be None if no checkpoint specified
    #
    #     # Verify base properties are available
    #     assert hasattr(settings, 'is_training')
    #     assert hasattr(settings, 'has_data_config')
    #
    #     # Test property behavior
    #     assert settings.is_training is True  # Because SESSION.inference = false
    #     assert settings.has_data_config is True

    def test_training_settings_has_correct_interface(self, config_file):
        """Training settings should implement the training protocol."""
        settings = load_settings(config_file)

        # Verify training-specific properties
        assert hasattr(settings, "mlflow_enabled")
        assert hasattr(settings, "has_training_config")

        # Verify base properties are available
        assert hasattr(settings, "is_training")
        assert hasattr(settings, "has_data_config")

        # Test property behavior
        assert settings.mlflow_enabled is False  # Because MLFLOW.enabled = false
        assert settings.has_training_config is True
        assert settings.is_training is True
        assert settings.has_data_config is True

    def test_load_sections_flexibility(self, config_file):
        """Test that load_sections allows arbitrary section combinations."""
        # Test loading only MODEL and DATASET sections
        settings = load_sections(config_file, ["MODEL", "DATASET"])

        # Should have requested sections
        assert hasattr(settings, "MODEL")
        assert hasattr(settings, "DATASET")
        assert settings.MODEL is not None
        assert settings.DATASET is not None

        # Should not have other sections in this case (they'll be None)
        # This demonstrates the flexibility of load_sections

    def test_load_sections_with_strict_mode(self, tmp_path: Path):
        """Test that strict mode works correctly."""
        # Create minimal config without TRAINING section
        minimal_config = """
[SESSION]
name = "test"

[MODEL]
name = "TestModel"

[DATAMODULE]
name = "TestDataModule"

[DATASET]
name = "TestDataset"
"""

        config_path = tmp_path / "minimal.toml"
        config_path.write_text(minimal_config)

        # Should work with non-strict mode
        settings = load_sections(config_path, ["MODEL", "DATASET"], strict=False)
        assert settings.MODEL is not None
        assert settings.DATASET is not None

        # Should fail with strict mode when requesting missing sections
        with pytest.raises(ValueError, match="Required sections missing"):
            load_sections(config_path, ["MODEL", "DATASET", "TRAINING"], strict=True)
