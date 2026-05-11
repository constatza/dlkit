"""Tests for VanillaExecutor LR tuning integration."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig


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
    mock_model.lr = 0.001

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
        """Training proceeds normally without LR tuner; lr is not changed."""
        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                executor.execute(mock_components, settings_without_lr_tuner)

        mock_components.trainer.fit.assert_called_once()
        assert mock_components.model.lr == 0.001

    def test_execute_with_lr_tuner_enabled(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """LR tuner is called and model.lr is updated to the suggested value."""
        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.005

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings_with_lr_tuner)

        mock_lr_tuner.tune.assert_called_once()
        call_args = mock_lr_tuner.tune.call_args
        assert call_args[0][0] == mock_components.trainer
        assert call_args[0][1] == mock_components.model
        assert call_args[0][2].min_lr == 1e-6
        assert call_args[0][2].max_lr == 0.1
        assert call_args[0][2].num_training == 50

        assert mock_components.model.lr == 0.005
        mock_components.trainer.fit.assert_called_once()

    def test_execute_with_lr_tuner_empty_config(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """Empty [TRAINING.lr_tuner] section creates LRTunerSettings with defaults."""
        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

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

        mock_lr_tuner.tune.assert_called_once()
        assert mock_components.model.lr == 0.003

    def test_execute_lr_tuner_failure_continues_training(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Training continues if LR tuner raises; original lr is preserved."""
        executor = VanillaExecutor()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.side_effect = RuntimeError("Tuning failed")

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings_with_lr_tuner)

        mock_components.trainer.fit.assert_called_once()
        assert mock_components.model.lr == 0.001

    def test_apply_lr_tuning_sets_model_lr(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """_apply_lr_tuning sets model.lr to the suggested value."""
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

        assert mock_components.model.lr == 0.007

    def test_apply_lr_tuning_skips_when_no_lr_tuner_setting(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """_apply_lr_tuning skips when TRAINING section has no lr_tuner."""
        executor = VanillaExecutor()

        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        settings = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings(),
        )

        mock_lr_tuner = Mock()

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
                settings,
            )

        mock_lr_tuner.tune.assert_not_called()

    def test_apply_lr_tuning_model_without_lr_skips_update(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """When model has no lr attribute, tuner still runs but update is skipped."""
        executor = VanillaExecutor()

        mock_model_no_lr = Mock(spec=[])
        components_no_lr = RuntimeComponents(
            model=mock_model_no_lr,
            datamodule=mock_components.datamodule,
            trainer=mock_components.trainer,
            shape_spec=None,
            meta={},
        )

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.008

        trainer = components_no_lr.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                trainer,
                components_no_lr.model,
                components_no_lr.datamodule,
                settings_with_lr_tuner,
            )

        # Tuner still ran, update was skipped gracefully
        mock_lr_tuner.tune.assert_called_once()
        assert not hasattr(components_no_lr.model, "lr")

    def test_apply_lr_tuning_noop_setter_leaves_lr_unchanged(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """When model.lr setter is a no-op, model.lr stays at original value after tuning."""
        executor = VanillaExecutor()

        class _NoOpLRModel:
            """Model whose lr setter is intentionally a no-op (like the old broken design)."""

            _lr: float = 0.001

            @property
            def lr(self) -> float:
                return self._lr

            @lr.setter
            def lr(self, value: float) -> None:
                pass  # no-op: does not store value

        noop_model = _NoOpLRModel()

        mock_lr_tuner = Mock()
        mock_lr_tuner.tune.return_value = 0.05

        trainer = mock_components.trainer
        assert trainer is not None
        with patch(
            "dlkit.engine.training.tuning.LRTuner",
            return_value=mock_lr_tuner,
        ):
            executor._apply_lr_tuning(
                trainer,
                noop_model,  # type: ignore[arg-type]
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        mock_lr_tuner.tune.assert_called_once()
        # lr was NOT updated because the setter is a no-op
        assert noop_model.lr == 0.001
        assert noop_model.lr != 0.05
