"""Tests for learning rate tuner."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from dlkit.engine.training.tuning import LRTuner
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings


@pytest.fixture
def lr_tuner_settings() -> LRTunerSettings:
    """Default LR tuner settings for testing."""
    return LRTunerSettings()


@pytest.fixture
def custom_lr_tuner_settings() -> LRTunerSettings:
    """Custom LR tuner settings for testing."""
    return LRTunerSettings(
        min_lr=1e-6,
        max_lr=0.1,
        num_training=50,
        mode="linear",
        early_stop_threshold=3.0,
    )


@pytest.fixture
def mock_trainer() -> Mock:
    """Mock PyTorch Lightning trainer."""
    return Mock()


@pytest.fixture
def mock_model() -> Mock:
    """Mock Lightning module."""
    return Mock()


@pytest.fixture
def mock_datamodule() -> Mock:
    """Mock Lightning datamodule."""
    return Mock()


class TestLRTunerSettings:
    """Test LRTunerSettings configuration class."""

    def test_default_values(self) -> None:
        """Test that default values are sensible."""
        settings = LRTunerSettings()

        assert settings.min_lr == 1e-8
        assert settings.max_lr == 1.0
        assert settings.num_training == 30  # Updated to match current default
        assert settings.mode == "exponential"
        assert settings.early_stop_threshold == 4.0

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        settings = LRTunerSettings(
            min_lr=1e-5,
            max_lr=0.01,
            num_training=200,
            mode="linear",
            early_stop_threshold=None,
        )

        assert settings.min_lr == 1e-5
        assert settings.max_lr == 0.01
        assert settings.num_training == 200
        assert settings.mode == "linear"
        assert settings.early_stop_threshold is None

    def test_mode_validation(self) -> None:
        """Test that mode only accepts valid values."""
        # Valid modes
        LRTunerSettings(mode="exponential")
        LRTunerSettings(mode="linear")

        # Invalid mode should raise validation error
        with pytest.raises(Exception):  # Pydantic ValidationError
            LRTunerSettings(mode=cast(Any, "invalid"))


class TestLRTuner:
    """Test LRTuner service class."""

    def test_tune_success(
        self,
        mock_trainer: Mock,
        mock_model: Mock,
        mock_datamodule: Mock,
        lr_tuner_settings: LRTunerSettings,
    ) -> None:
        """Test successful LR tuning."""
        # Setup mock lr_finder
        mock_lr_finder = Mock()
        mock_lr_finder.suggestion.return_value = 0.001

        # Mock Tuner class
        with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
            mock_tuner_instance = Mock()
            mock_tuner_instance.lr_find.return_value = mock_lr_finder
            MockTuner.return_value = mock_tuner_instance

            # Run tuner
            tuner = LRTuner()
            result = tuner.tune(mock_trainer, mock_model, lr_tuner_settings, mock_datamodule)

            # Verify result
            assert result == 0.001

            # Verify Tuner was created with trainer
            MockTuner.assert_called_once_with(mock_trainer)

            # Verify lr_find was called with correct parameters
            mock_tuner_instance.lr_find.assert_called_once_with(
                mock_model,
                datamodule=mock_datamodule,
                min_lr=lr_tuner_settings.min_lr,
                max_lr=lr_tuner_settings.max_lr,
                num_training=lr_tuner_settings.num_training,
                mode=lr_tuner_settings.mode,
                early_stop_threshold=lr_tuner_settings.early_stop_threshold,
            )

    def test_tune_with_custom_settings(
        self,
        mock_trainer: Mock,
        mock_model: Mock,
        custom_lr_tuner_settings: LRTunerSettings,
    ) -> None:
        """Test tuning with custom settings."""
        mock_lr_finder = Mock()
        mock_lr_finder.suggestion.return_value = 0.005

        with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
            mock_tuner_instance = Mock()
            mock_tuner_instance.lr_find.return_value = mock_lr_finder
            MockTuner.return_value = mock_tuner_instance

            tuner = LRTuner()
            result = tuner.tune(mock_trainer, mock_model, custom_lr_tuner_settings, None)

            assert result == 0.005

            # Verify custom settings were used
            mock_tuner_instance.lr_find.assert_called_once_with(
                mock_model,
                datamodule=None,
                min_lr=1e-6,
                max_lr=0.1,
                num_training=50,
                mode="linear",
                early_stop_threshold=3.0,
            )

    def test_tune_no_suggestion_raises_error(
        self,
        mock_trainer: Mock,
        mock_model: Mock,
        lr_tuner_settings: LRTunerSettings,
    ) -> None:
        """Test that None suggestion raises RuntimeError."""
        mock_lr_finder = Mock()
        mock_lr_finder.suggestion.return_value = None

        with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
            mock_tuner_instance = Mock()
            mock_tuner_instance.lr_find.return_value = mock_lr_finder
            MockTuner.return_value = mock_tuner_instance

            tuner = LRTuner()

            with pytest.raises(RuntimeError, match="failed to suggest a learning rate"):
                tuner.tune(mock_trainer, mock_model, lr_tuner_settings, None)

    def test_tune_without_datamodule(
        self,
        mock_trainer: Mock,
        mock_model: Mock,
        lr_tuner_settings: LRTunerSettings,
    ) -> None:
        """Test tuning works without datamodule (None)."""
        mock_lr_finder = Mock()
        mock_lr_finder.suggestion.return_value = 0.002

        with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
            mock_tuner_instance = Mock()
            mock_tuner_instance.lr_find.return_value = mock_lr_finder
            MockTuner.return_value = mock_tuner_instance

            tuner = LRTuner()
            result = tuner.tune(mock_trainer, mock_model, lr_tuner_settings, None)

            assert result == 0.002

            # Verify datamodule=None was passed
            call_kwargs = mock_tuner_instance.lr_find.call_args[1]
            assert call_kwargs["datamodule"] is None
