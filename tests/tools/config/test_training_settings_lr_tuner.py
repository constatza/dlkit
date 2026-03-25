"""Tests for TrainingSettings with LR tuner configuration."""

from __future__ import annotations

from dlkit.tools.config.lr_tuner_settings import LRTunerSettings
from dlkit.tools.config.training_settings import TrainingSettings


class TestTrainingSettingsLRTuner:
    """Test TrainingSettings LR tuner configuration."""

    def test_default_lr_tuner_is_none(self) -> None:
        """Test that lr_tuner defaults to None (disabled)."""
        settings = TrainingSettings()

        assert settings.lr_tuner is None

    def test_lr_tuner_with_default_settings(self) -> None:
        """Test configuring lr_tuner with default LRTunerSettings."""
        settings = TrainingSettings(lr_tuner=LRTunerSettings())

        assert settings.lr_tuner is not None
        assert settings.lr_tuner.min_lr == 1e-8
        assert settings.lr_tuner.max_lr == 1.0
        assert settings.lr_tuner.num_training == 30  # Updated to match current default
        assert settings.lr_tuner.mode == "exponential"
        assert settings.lr_tuner.early_stop_threshold == 4.0

    def test_lr_tuner_with_custom_settings(self) -> None:
        """Test configuring lr_tuner with custom settings."""
        settings = TrainingSettings(
            lr_tuner=LRTunerSettings(
                min_lr=1e-6,
                max_lr=0.1,
                num_training=50,
                mode="linear",
                early_stop_threshold=3.0,
            )
        )

        assert settings.lr_tuner is not None
        assert settings.lr_tuner.min_lr == 1e-6
        assert settings.lr_tuner.max_lr == 0.1
        assert settings.lr_tuner.num_training == 50
        assert settings.lr_tuner.mode == "linear"
        assert settings.lr_tuner.early_stop_threshold == 3.0

    def test_lr_tuner_with_empty_dict(self) -> None:
        """Test configuring lr_tuner with empty dict (simulates empty TOML section)."""
        settings = TrainingSettings.model_validate({"lr_tuner": {}})

        assert settings.lr_tuner is not None
        # Should use all defaults
        assert settings.lr_tuner.min_lr == 1e-8
        assert settings.lr_tuner.max_lr == 1.0
        assert settings.lr_tuner.num_training == 30  # Updated to match current default
        assert settings.lr_tuner.mode == "exponential"

    def test_lr_tuner_from_dict(self) -> None:
        """Test creating TrainingSettings with lr_tuner from dict."""
        config = {
            "lr_tuner": {
                "min_lr": 1e-5,
                "max_lr": 0.05,
                "num_training": 75,
                "mode": "linear",
            }
        }

        settings = TrainingSettings.model_validate(config)

        assert settings.lr_tuner is not None
        assert settings.lr_tuner.min_lr == 1e-5
        assert settings.lr_tuner.max_lr == 0.05
        assert settings.lr_tuner.num_training == 75
        assert settings.lr_tuner.mode == "linear"

    def test_lr_tuner_serialization(self) -> None:
        """Test that lr_tuner can be serialized and deserialized."""
        original = TrainingSettings(
            lr_tuner=LRTunerSettings(
                min_lr=1e-7,
                max_lr=0.2,
                num_training=150,
            )
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back
        restored = TrainingSettings.model_validate(data)

        assert restored.lr_tuner is not None
        assert restored.lr_tuner.min_lr == 1e-7
        assert restored.lr_tuner.max_lr == 0.2
        assert restored.lr_tuner.num_training == 150

    def test_model_copy_preserves_lr_tuner(self) -> None:
        """Test that model_copy preserves lr_tuner settings."""
        original = TrainingSettings(
            epochs=50,
            lr_tuner=LRTunerSettings(min_lr=1e-6, max_lr=0.1),
        )

        # Create copy with updated epochs
        copy = original.model_copy(update={"epochs": 100})

        assert copy.epochs == 100
        assert copy.lr_tuner is not None
        assert copy.lr_tuner.min_lr == 1e-6
        assert copy.lr_tuner.max_lr == 0.1

    def test_lr_tuner_with_early_stop_disabled(self) -> None:
        """Test lr_tuner with early_stop_threshold set to None."""
        settings = TrainingSettings(lr_tuner=LRTunerSettings(early_stop_threshold=None))

        assert settings.lr_tuner is not None
        assert settings.lr_tuner.early_stop_threshold is None

    def test_integration_with_other_training_settings(self) -> None:
        """Test that lr_tuner works alongside other training settings."""
        settings = TrainingSettings(
            epochs=200,
            patience=20,
            monitor_metric="val_accuracy",
            mode="max",
            lr_tuner=LRTunerSettings(
                min_lr=1e-5,
                max_lr=0.01,
            ),
        )

        # Verify training settings
        assert settings.epochs == 200
        assert settings.patience == 20
        assert settings.monitor_metric == "val_accuracy"
        assert settings.mode == "max"

        # Verify lr_tuner settings
        assert settings.lr_tuner is not None
        assert settings.lr_tuner.min_lr == 1e-5
        assert settings.lr_tuner.max_lr == 0.01
