"""Integration test demonstrating proper override behavior.

This test shows the complete behavior ensuring:
1. Overrides are None by default
2. Settings values take precedence when no overrides are specified
3. Override system works correctly when values are provided
"""

from __future__ import annotations

import pytest

from dlkit.interfaces.api.commands.train_command import TrainCommand, TrainCommandInput
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.dataloader_settings import DataloaderSettings
from dlkit.tools.config.datamodule_settings import DataModuleSettings
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.training_settings import TrainingSettings


def _require_training(settings: GeneralSettings) -> TrainingSettings:
    """Narrow TRAINING to the configured TrainingSettings instance."""
    training = settings.TRAINING
    assert training is not None
    return training


def _require_datamodule(settings: GeneralSettings) -> DataModuleSettings:
    """Narrow DATAMODULE to the configured DataModuleSettings instance."""
    datamodule = settings.DATAMODULE
    assert datamodule is not None
    return datamodule


def _require_numeric_lr(settings: TrainingSettings) -> int | float:
    """Narrow optimizer.lr to a concrete numeric value for assertions."""
    lr = settings.optimizer.lr
    assert isinstance(lr, int | float)
    return lr


@pytest.fixture
def sample_settings() -> GeneralSettings:
    """Create sample settings for integration test."""
    return GeneralSettings(
        SESSION=SessionSettings(inference=False),
        TRAINING=TrainingSettings(epochs=50),
        DATAMODULE=DataModuleSettings(dataloader=DataloaderSettings(batch_size=16)),
    )


class TestOverrideIntegration:
    """Integration tests for the complete override behavior."""

    def test_command_input_defaults_are_none(self) -> None:
        """Test that TrainCommandInput has None defaults for all override fields."""
        # Create command input with no arguments - all should default to None
        input_data = TrainCommandInput()

        # Verify all override fields are None by default
        assert input_data.checkpoint_path is None
        assert input_data.epochs is None
        assert input_data.batch_size is None
        assert input_data.learning_rate is None
        assert input_data.experiment_name is None
        assert input_data.run_name is None
        assert input_data.additional_overrides == {}

    def test_settings_values_preserved_when_no_overrides(
        self, sample_settings: GeneralSettings
    ) -> None:
        """Test that settings values are preserved when no overrides are provided."""
        train_command = TrainCommand()

        # Create input with no overrides (all None)
        input_data = TrainCommandInput()

        # Build overrides dict - should be empty (all None inputs)
        overrides = train_command._build_overrides_dict(input_data)
        assert overrides == {}

        # Apply empty overrides - should preserve original settings
        result = train_command.override_manager.apply_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_settings)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        # Original values should be preserved
        assert result_training.epochs == sample_training.epochs  # 50
        assert (
            result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size
        )  # 16

    def test_overrides_applied_when_values_provided(self, sample_settings: GeneralSettings) -> None:
        """Test that overrides are applied when values are provided."""
        train_command = TrainCommand()

        # Create input with some overrides provided
        input_data = TrainCommandInput(
            epochs=100,  # Override from default 50
            batch_size=64,  # Override from default 16
            learning_rate=0.01,  # New value
        )

        # Build overrides dict - should only contain non-None values
        overrides = train_command._build_overrides_dict(input_data)
        expected_overrides = {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.01,
        }
        assert overrides == expected_overrides

        # Apply overrides
        result = train_command.override_manager.apply_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        result_datamodule = _require_datamodule(result)

        # Overridden values should be updated
        assert result_training.epochs == 100  # Changed from 50
        assert result_datamodule.dataloader.batch_size == 64  # Changed from 16
        assert float(_require_numeric_lr(result_training)) == pytest.approx(0.01)  # New value

    def test_partial_overrides_preserve_non_overridden_values(
        self, sample_settings: GeneralSettings
    ) -> None:
        """Test that partial overrides preserve non-overridden values."""
        train_command = TrainCommand()

        # Create input with only some fields overridden
        input_data = TrainCommandInput(
            epochs=200,  # Only override epochs
            # All other fields remain None
        )

        overrides = train_command._build_overrides_dict(input_data)
        assert overrides == {"epochs": 200}

        result = train_command.override_manager.apply_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        # Only epochs should be changed
        assert result_training.epochs == 200  # Changed
        assert (
            result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size
        )  # Preserved (16)

    def test_none_values_explicitly_ignored(self, sample_settings: GeneralSettings) -> None:
        """Test that explicitly passing None values are ignored."""
        train_command = TrainCommand()

        # Create input with explicit None values
        input_data = TrainCommandInput(
            epochs=None,  # Explicitly None
            batch_size=None,  # Explicitly None
            learning_rate=50,  # Actual override value
        )

        overrides = train_command._build_overrides_dict(input_data)
        assert overrides == {"learning_rate": 50}

        result = train_command.override_manager.apply_overrides(sample_settings, **overrides)
        result_training = _require_training(result)
        sample_training = _require_training(sample_settings)
        result_datamodule = _require_datamodule(result)
        sample_datamodule = _require_datamodule(sample_settings)

        # Only learning_rate should be changed, None values ignored
        assert result_training.epochs == sample_training.epochs  # Preserved
        assert (
            result_datamodule.dataloader.batch_size == sample_datamodule.dataloader.batch_size
        )  # Preserved
        assert float(_require_numeric_lr(result_training)) == pytest.approx(50)  # Changed


def test_override_workflow_summary() -> None:
    """Summary test showing the complete override workflow."""
    input_data = TrainCommandInput()
    assert input_data.epochs is None
    assert input_data.batch_size is None
    assert input_data.learning_rate is None
    assert input_data.additional_overrides == {}
