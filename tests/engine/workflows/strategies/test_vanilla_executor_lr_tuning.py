"""Tests for VanillaExecutor LR tuning integration."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.vanilla_executor import VanillaExecutor
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    LBFGSSettings,
    StepLRSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
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
            ),
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

        trainer = cast("Any", mock_components.trainer)
        trainer.fit.assert_called_once()
        assert mock_components.model.lr == 0.001

    def test_execute_with_lr_tuner_enabled(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """LR tuner is called and model.lr is updated to the suggested value."""
        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch.object(
                    executor, "_find_lr_with_projected_policy", return_value=0.005
                ) as find_lr:
                    executor.execute(mock_components, settings_with_lr_tuner)

        find_lr.assert_called_once()
        assert mock_components.model.lr == 0.005
        cast("Any", mock_components.trainer).fit.assert_called_once()

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

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch.object(executor, "_find_lr_with_projected_policy", return_value=0.003):
                    executor.execute(mock_components, settings_with_empty_lr_tuner)

        assert mock_components.model.lr == 0.003

    def test_execute_lr_tuner_failure_continues_training(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Training continues if LR tuner raises; original lr is preserved."""
        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch.object(
                    executor,
                    "_find_lr_with_projected_policy",
                    side_effect=RuntimeError("Tuning failed"),
                ):
                    executor.execute(mock_components, settings_with_lr_tuner)

        cast("Any", mock_components.trainer).fit.assert_called_once()
        assert mock_components.model.lr == 0.001

    def test_apply_lr_tuning_sets_model_lr(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """_apply_lr_tuning sets model.lr to the suggested value."""
        executor = VanillaExecutor()

        with patch.object(executor, "_find_lr_with_projected_policy", return_value=0.007):
            executor._apply_lr_tuning(
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
        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        executor = VanillaExecutor()
        settings = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings(),
        )

        with patch.object(executor, "_find_lr_with_projected_policy") as find_lr:
            executor._apply_lr_tuning(
                mock_components.model,
                mock_components.datamodule,
                settings,
            )

        find_lr.assert_not_called()

    def test_apply_lr_tuning_model_without_lr_skips_update(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """When model does not implement ILRTunable, LR tuning is skipped entirely."""
        executor = VanillaExecutor()
        mock_model_no_lr = Mock(spec=[])

        with patch.object(executor, "_find_lr_with_projected_policy") as find_lr:
            executor._apply_lr_tuning(
                mock_model_no_lr,
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        find_lr.assert_not_called()
        assert not hasattr(mock_model_no_lr, "lr")

    def test_apply_lr_tuning_skips_for_lbfgs(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """LR tuner is not called when the configured optimizer requires a closure (LBFGS)."""
        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        settings = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings(
                lr_tuner=LRTunerSettings(min_lr=1e-5, max_lr=0.1, num_training=100),
                optimizer=OptimizerPolicySettings(default_optimizer=LBFGSSettings()),
            ),
        )

        executor = VanillaExecutor()
        mock_lr_tuner = Mock()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch(
                    "dlkit.engine.training.tuning.LRTuner",
                    return_value=mock_lr_tuner,
                ):
                    executor.execute(mock_components, settings)

        mock_lr_tuner.tune.assert_not_called()
        cast("Any", mock_components.trainer).fit.assert_called_once()

    def test_apply_lr_tuning_projects_multi_stage_policy_to_temporary_wrapper(
        self,
        mock_components: RuntimeComponents,
    ) -> None:
        """Sequential staged policies should tune through a temporary stage-0 projection."""
        from dlkit.infrastructure.config.session_settings import SessionSettings
        from dlkit.infrastructure.config.training_settings import TrainingSettings

        settings = TrainingWorkflowConfig(
            SESSION=SessionSettings(workflow="train", seed=42),
            TRAINING=TrainingSettings(
                lr_tuner=LRTunerSettings(min_lr=1e-5, max_lr=0.1, num_training=100),
                optimizer=OptimizerPolicySettings(
                    stages=(
                        OptimizationStageSettings(
                            optimizer=AdamWSettings(lr=1e-3),
                            scheduler=StepLRSettings(step_size=1, gamma=0.5),
                            trigger=TriggerSettings(at_epoch=5),
                        ),
                        OptimizationStageSettings(
                            optimizer=AdamSettings(lr=1e-3),
                            scheduler=StepLRSettings(step_size=1, gamma=0.5),
                        ),
                    )
                ),
            ),
        )

        executor = VanillaExecutor()

        with patch("pytorch_lightning.seed_everything"):
            with patch("dlkit.infrastructure.precision.service.get_precision_service"):
                with patch.object(
                    executor, "_find_lr_with_projected_policy", return_value=0.02
                ) as find_lr:
                    executor.execute(mock_components, settings)

        find_lr.assert_called_once()
        assert mock_components.model.lr == 0.02
        cast("Any", mock_components.trainer).fit.assert_called_once()

    def test_apply_lr_tuning_noop_setter_leaves_lr_unchanged(
        self,
        mock_components: RuntimeComponents,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """When model.lr setter is a no-op, model.lr stays at original value after tuning."""
        executor = VanillaExecutor()

        from lightning.pytorch import LightningModule

        class _NoOpLRModel(LightningModule):
            """Model whose lr setter is intentionally a no-op."""

            _lr: float = 0.001

            @property
            def lr(self) -> float:
                return self._lr

            @lr.setter
            def lr(self, value: float) -> None:
                pass

        noop_model = _NoOpLRModel()

        with patch.object(executor, "_find_lr_with_projected_policy", return_value=0.05):
            executor._apply_lr_tuning(
                cast("Any", noop_model),
                mock_components.datamodule,
                settings_with_lr_tuner,
            )

        assert noop_model.lr == 0.001
        assert noop_model.lr != 0.05

    def test_find_lr_restores_controller_after_success(
        self,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Original controller and automatic_optimization are restored after LR finding."""
        import torch.nn as nn

        from dlkit.engine.training.tuning.plans import SupportedLRTuningPlan
        from dlkit.infrastructure.config.optimizer_component import AdamWSettings
        from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
        from dlkit.infrastructure.config.trainer_settings import TrainerSettings

        executor = VanillaExecutor()
        original_controller = MagicMock()
        original_controller.requires_manual_optimization = True

        model = MagicMock()
        model.model = nn.Linear(2, 2)
        model._optimization_controller = original_controller
        model.automatic_optimization = False

        tuning_plan = SupportedLRTuningPlan(
            projected_policy=OptimizerPolicySettings(default_optimizer=AdamWSettings(lr=1e-3)),
        )

        fake_tuning_controller = MagicMock()
        fake_tuning_controller.requires_manual_optimization = False

        with patch(
            "dlkit.engine.training.optimization.controllers.build_optimization_controller",
            return_value=fake_tuning_controller,
        ):
            with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
                mock_finder = MagicMock()
                mock_finder.suggestion.return_value = 0.01
                MockTuner.return_value.lr_find.return_value = mock_finder
                with patch.object(TrainerSettings, "build", return_value=MagicMock()):
                    result = executor._find_lr_with_projected_policy(
                        model,
                        None,
                        settings_with_lr_tuner,
                        tuning_plan,
                        settings_with_lr_tuner.TRAINING.lr_tuner,
                    )

        assert result == 0.01
        assert model._optimization_controller is original_controller
        assert model.automatic_optimization is False

    def test_find_lr_restores_controller_after_failure(
        self,
        settings_with_lr_tuner: TrainingWorkflowConfig,
    ) -> None:
        """Original controller is restored even when LR finding raises."""
        import torch.nn as nn

        from dlkit.engine.training.tuning.plans import SupportedLRTuningPlan
        from dlkit.infrastructure.config.optimizer_component import AdamWSettings
        from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
        from dlkit.infrastructure.config.trainer_settings import TrainerSettings

        executor = VanillaExecutor()
        original_controller = MagicMock()
        model = MagicMock()
        model.model = nn.Linear(2, 2)
        model._optimization_controller = original_controller
        model.automatic_optimization = True

        tuning_plan = SupportedLRTuningPlan(
            projected_policy=OptimizerPolicySettings(default_optimizer=AdamWSettings(lr=1e-3)),
        )

        fake_tuning_controller = MagicMock()
        fake_tuning_controller.requires_manual_optimization = True

        with patch(
            "dlkit.engine.training.optimization.controllers.build_optimization_controller",
            return_value=fake_tuning_controller,
        ):
            with patch("dlkit.engine.training.tuning.lr_tuner.Tuner") as MockTuner:
                MockTuner.return_value.lr_find.side_effect = RuntimeError("tuner exploded")
                with patch.object(TrainerSettings, "build", return_value=MagicMock()):
                    with pytest.raises(RuntimeError):
                        executor._find_lr_with_projected_policy(
                            model,
                            None,
                            settings_with_lr_tuner,
                            tuning_plan,
                            settings_with_lr_tuner.TRAINING.lr_tuner,
                        )

        assert model._optimization_controller is original_controller
        assert model.automatic_optimization is True
