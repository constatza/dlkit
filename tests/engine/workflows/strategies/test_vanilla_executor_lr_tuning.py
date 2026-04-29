"""Tests for VanillaExecutor LR tuning integration."""

from __future__ import annotations

from typing import cast
from unittest.mock import Mock, patch

import pytest

from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.optimizer_settings import OptimizerSettings
from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig


def _trainer_mock(components: RuntimeComponents) -> Mock:
    trainer = components.trainer
    assert trainer is not None
    return cast(Mock, trainer)


def _optimizer_settings(components: RuntimeComponents) -> OptimizerSettings:
    optimizer = components.model.optimizer
    assert isinstance(optimizer, OptimizerSettings)
    return optimizer


@pytest.fixture
def mock_components() -> RuntimeComponents:
    """Mock build components for testing."""
    mock_trainer = Mock()
    mock_trainer.fit = Mock()
    mock_trainer.predict = Mock()
    mock_trainer.test = Mock()
    mock_trainer.callback_metrics = {}
    mock_trainer.progress_bar_metrics = {}
    mock_trainer.logged_metrics = {}
    mock_trainer.callbacks = []

    mock_model = Mock()
    mock_model.optimizer = OptimizerSettings(lr=0.001)  # type: ignore[call-arg]

    mock_datamodule = Mock()

    return RuntimeComponents(
        model=mock_model,
        datamodule=mock_datamodule,
        trainer=mock_trainer,
        shape_spec=None,
        meta={},
    )


@pytest.fixture
def settings_without_lr_tuner() -> TrainingWorkflowConfig:
    """Settings without LR tuner configured."""
    from dlkit.infrastructure.config.session_settings import SessionSettings
    from dlkit.infrastructure.config.training_settings import TrainingSettings

    return TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train", seed=42),
        TRAINING=TrainingSettings(),
    )


@pytest.fixture
def settings_with_lr_tuner() -> TrainingWorkflowConfig:
    """Settings with LR tuner configured (enabled)."""
    from dlkit.infrastructure.config.session_settings import SessionSettings
    from dlkit.infrastructure.config.training_settings import TrainingSettings

    return TrainingWorkflowConfig(
        SESSION=SessionSettings(workflow="train", seed=42),
        TRAINING=TrainingSettings(
            lr_tuner=LRTunerSettings(
                min_lr=1e-6,
                max_lr=0.1,
                num_training=50,
            )
        ),
    )


class TestVanillaExecutorLRTuning:
    """Test VanillaExecutor LR tuning integration."""

    def test_execute_without_lr_tuner_configured(
        self,
        mock_components: RuntimeComponents,
        settings_without_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test that training proceeds normally without LR tuner."""
        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                executor.execute(mock_components, settings_without_lr_tuner)

        # Verify training was called
        _trainer_mock(mock_components).fit.assert_called_once()

        # Verify original LR was NOT modified
        assert _optimizer_settings(mock_components).lr == 0.001

    def test_execute_with_lr_tuner_enabled(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test that LR tuner is called and updates learning rate."""
        executor = VanillaExecutor()

        # Mock the LRTuner
        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.005

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings_with_lr_tuner)

        # Verify LR tuner was called
        mock_lr_tuner.tune.assert_called_once()

        # Verify tuner was called with correct arguments
        call_args = mock_lr_tuner.tune.call_args
        assert call_args[0][0] == mock_components.trainer
        assert call_args[0][1] == mock_components.model
        assert call_args[0][2].min_lr == 1e-6
        assert call_args[0][2].max_lr == 0.1
        assert call_args[0][2].num_training == 50

        # Verify learning rate was updated
        assert _optimizer_settings(mock_components).lr == 0.005

        # Verify training was called after tuning
        _trainer_mock(mock_components).fit.assert_called_once()

    def test_execute_with_lr_tuner_empty_config(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """Test that LR tuner is called with empty dict config (all defaults)."""
        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        # Simulate empty TOML section: [TRAINING.lr_tuner]
        settings_with_empty_lr_tuner = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings.model_validate({"lr_tuner": {}}),
        )

        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.003

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings_with_empty_lr_tuner)

        # Verify LR tuner WAS called (empty dict creates LRTunerSettings with defaults)
        mock_lr_tuner.tune.assert_called_once()

        # Verify LR was updated
        assert _optimizer_settings(mock_components).lr == 0.003

    def test_execute_lr_tuner_failure_continues_training(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test that training continues if LR tuner fails."""
        executor = VanillaExecutor()

        # Mock LRTuner to raise exception
        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.side_effect = RuntimeError("Tuning failed")

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    # Should not raise - graceful degradation
                    executor.execute(mock_components, settings_with_lr_tuner)

        # Verify training still proceeded
        _trainer_mock(mock_components).fit.assert_called_once()

        # Verify original LR was preserved
        assert _optimizer_settings(mock_components).lr == 0.001

    def test_apply_lr_tuning_helper_method(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test _apply_lr_tuning helper method in isolation."""
        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.007

        trainer = mock_components.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                trainer,
                mock_components.model,
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        # Verify LR was updated
        assert _optimizer_settings(mock_components).lr == 0.007

    def test_apply_lr_tuning_with_no_training_settings(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """Test _apply_lr_tuning when TRAINING without lr_tuner configured."""
        executor = VanillaExecutor()

        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        settings = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings(),  # No lr_tuner configured
        )

        mock_lr_tuner = Mock()

        trainer = mock_components.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            # Should not raise
            executor._apply_lr_tuning(
                trainer,
                mock_components.model,
                mock_components.datamodule,
                settings,
            )

        # Verify tuner was not called
        mock_lr_tuner.tune.assert_not_called()

    def test_apply_lr_tuning_model_without_optimizer_attribute(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test graceful handling when model lacks optimizer attribute."""
        executor = VanillaExecutor()

        # Model without optimizer attribute
        mock_model_no_opt = Mock(spec=[])
        components_no_opt = RuntimeComponents(
            model=mock_model_no_opt,
            datamodule=mock_components.datamodule,
            trainer=mock_components.trainer,
            shape_spec=None,
            meta={},
        )

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.008

        trainer = components_no_opt.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            # Should not raise - logs warning instead
            executor._apply_lr_tuning(
                trainer,
                components_no_opt.model,
                components_no_opt.datamodule,
                settings_with_lr_tuner,
            )

        # Verify tuner was still called
        mock_lr_tuner.tune.assert_called_once()

    def test_mutable_optimizer_update(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Test that optimizer update uses mutable update_settings.

        With frozen=True, update_settings returns a new OptimizerSettings instance
        which is assigned back to model.optimizer.
        """
        executor = VanillaExecutor()

        original_optimizer = _optimizer_settings(mock_components)
        original_lr = original_optimizer.lr

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.009

        trainer = mock_components.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                trainer,
                mock_components.model,
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        # Verify new LR is set on the (new) optimizer instance
        assert _optimizer_settings(mock_components).lr == 0.009

        # Original optimizer object is unchanged (immutable semantics)
        assert original_optimizer.lr == original_lr
