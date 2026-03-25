"""Tests for VanillaExecutor LR tuning integration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.runtime.workflows.strategies.core.vanilla_executor import VanillaExecutor
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.lr_tuner_settings import LRTunerSettings
from dlkit.tools.config.optimizer_settings import OptimizerSettings


@pytest.fixture
def mock_components() -> BuildComponents:
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
    mock_model.optimizer = OptimizerSettings(lr=0.001)

    mock_datamodule = Mock()

    return BuildComponents(
        model=mock_model,
        datamodule=mock_datamodule,
        trainer=mock_trainer,
        shape_spec=None,
        meta={},
    )


@pytest.fixture
def settings_without_lr_tuner() -> GeneralSettings:
    """Settings without LR tuner configured."""
    from dlkit.tools.config.session_settings import SessionSettings
    from dlkit.tools.config.training_settings import TrainingSettings

    return GeneralSettings(
        SESSION=SessionSettings(seed=42),
        TRAINING=TrainingSettings(),
    )


@pytest.fixture
def settings_with_lr_tuner() -> GeneralSettings:
    """Settings with LR tuner configured (enabled)."""
    from dlkit.tools.config.session_settings import SessionSettings
    from dlkit.tools.config.training_settings import TrainingSettings

    return GeneralSettings(
        SESSION=SessionSettings(seed=42),
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
        mock_components: BuildComponents,
        settings_without_lr_tuner: GeneralSettings,
    ) -> None:
        """Test that training proceeds normally without LR tuner."""
        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.interfaces.api.services.precision_service.get_precision_service"):
                executor.execute(mock_components, settings_without_lr_tuner)

        # Verify training was called
        mock_components.trainer.fit.assert_called_once()

        # Verify original LR was NOT modified
        assert mock_components.model.optimizer.lr == 0.001

    def test_execute_with_lr_tuner_enabled(
        self,
        mock_components: BuildComponents,
        settings_with_lr_tuner: GeneralSettings,
    ) -> None:
        """Test that LR tuner is called and updates learning rate."""
        executor = VanillaExecutor()

        # Mock the LRTuner
        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.005

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.interfaces.api.services.precision_service.get_precision_service"):
                with patch(
                    "dlkit.runtime.workflows.strategies.tuning.LRTuner",
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
        assert mock_components.model.optimizer.lr == 0.005

        # Verify training was called after tuning
        mock_components.trainer.fit.assert_called_once()

    def test_execute_with_lr_tuner_empty_config(
        self,
        mock_components: BuildComponents,
    ) -> None:
        """Test that LR tuner is called with empty dict config (all defaults)."""
        from dlkit.tools.config.session_settings import SessionSettings
        from dlkit.tools.config.training_settings import TrainingSettings

        # Simulate empty TOML section: [TRAINING.lr_tuner]
        settings_with_empty_lr_tuner = GeneralSettings(
            SESSION=SessionSettings(seed=42),
            TRAINING=TrainingSettings(lr_tuner={}),
        )

        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.003

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.interfaces.api.services.precision_service.get_precision_service"):
                with patch(
                    "dlkit.runtime.workflows.strategies.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings_with_empty_lr_tuner)

        # Verify LR tuner WAS called (empty dict creates LRTunerSettings with defaults)
        mock_lr_tuner.tune.assert_called_once()

        # Verify LR was updated
        assert mock_components.model.optimizer.lr == 0.003

    def test_execute_lr_tuner_failure_continues_training(
        self,
        mock_components: BuildComponents,
        settings_with_lr_tuner: GeneralSettings,
    ) -> None:
        """Test that training continues if LR tuner fails."""
        executor = VanillaExecutor()

        # Mock LRTuner to raise exception
        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.side_effect = RuntimeError("Tuning failed")

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.interfaces.api.services.precision_service.get_precision_service"):
                with patch(
                    "dlkit.runtime.workflows.strategies.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    # Should not raise - graceful degradation
                    executor.execute(mock_components, settings_with_lr_tuner)

        # Verify training still proceeded
        mock_components.trainer.fit.assert_called_once()

        # Verify original LR was preserved
        assert mock_components.model.optimizer.lr == 0.001

    def test_apply_lr_tuning_helper_method(
        self,
        mock_components: BuildComponents,
        settings_with_lr_tuner: GeneralSettings,
    ) -> None:
        """Test _apply_lr_tuning helper method in isolation."""
        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.007

        with patch(
            "dlkit.runtime.workflows.strategies.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                mock_components.trainer,
                mock_components.model,
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        # Verify LR was updated
        assert mock_components.model.optimizer.lr == 0.007

    def test_apply_lr_tuning_with_no_training_settings(
        self,
        mock_components: BuildComponents,
    ) -> None:
        """Test _apply_lr_tuning when TRAINING is None."""
        executor = VanillaExecutor()

        from dlkit.tools.config.session_settings import SessionSettings

        settings = GeneralSettings(SESSION=SessionSettings(seed=42), TRAINING=None)

        mock_lr_tuner = Mock()

        with patch(
            "dlkit.runtime.workflows.strategies.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            # Should not raise
            executor._apply_lr_tuning(
                mock_components.trainer,
                mock_components.model,
                mock_components.datamodule,
                settings,
            )

        # Verify tuner was not called
        mock_lr_tuner.tune.assert_not_called()

    def test_apply_lr_tuning_model_without_optimizer_attribute(
        self,
        mock_components: BuildComponents,
        settings_with_lr_tuner: GeneralSettings,
    ) -> None:
        """Test graceful handling when model lacks optimizer attribute."""
        executor = VanillaExecutor()

        # Model without optimizer attribute
        mock_model_no_opt = Mock(spec=[])
        components_no_opt = BuildComponents(
            model=mock_model_no_opt,
            datamodule=mock_components.datamodule,
            trainer=mock_components.trainer,
            shape_spec=None,
            meta={},
        )

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.008

        with patch(
            "dlkit.runtime.workflows.strategies.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            # Should not raise - logs warning instead
            executor._apply_lr_tuning(
                components_no_opt.trainer,
                components_no_opt.model,
                components_no_opt.datamodule,
                settings_with_lr_tuner,
            )

        # Verify tuner was still called
        mock_lr_tuner.tune.assert_called_once()

    def test_mutable_optimizer_update(
        self,
        mock_components: BuildComponents,
        settings_with_lr_tuner: GeneralSettings,
    ) -> None:
        """Test that optimizer update uses mutable update_settings.

        With frozen=True, update_settings returns a new OptimizerSettings instance
        which is assigned back to model.optimizer.
        """
        executor = VanillaExecutor()

        original_optimizer = mock_components.model.optimizer
        original_lr = original_optimizer.lr

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.009

        with patch(
            "dlkit.runtime.workflows.strategies.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                mock_components.trainer,
                mock_components.model,
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        # Verify new LR is set on the (new) optimizer instance
        assert mock_components.model.optimizer.lr == 0.009

        # Original optimizer object is unchanged (immutable semantics)
        assert original_optimizer.lr == original_lr
