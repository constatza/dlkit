"""Standard-wrapper control cases for staged optimization with LR tuning."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest

from dlkit.common import TrainingResult
from dlkit.engine.training.optimization.controllers import ManualOptimizationController
from dlkit.engine.training.tuning import LRTuner
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.lr_tuner_settings import LRTunerSettings
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    StepLRSettings,
)
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
from dlkit.infrastructure.config.trainer_settings import TrainerSettings
from dlkit.infrastructure.config.workflow_configs import TrainingWorkflowConfig
from dlkit.interfaces.api import train as api_train


def _build_standard_staged_settings(
    training_settings: TrainingWorkflowConfig,
) -> TrainingWorkflowConfig:
    """Create the staged standard-wrapper control scenario."""
    assert training_settings.TRAINING is not None
    training = training_settings.TRAINING.model_copy(
        update={
            "epochs": 3,
            "lr_tuner": LRTunerSettings(min_lr=1e-6, max_lr=0.1, num_training=20),
            "trainer": TrainerSettings.model_validate(
                {
                    "fast_dev_run": False,
                    "enable_checkpointing": False,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "limit_train_batches": 8,
                    "limit_val_batches": 2,
                    "max_epochs": 3,
                }
            ),
            "optimizer": OptimizerPolicySettings(
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
        }
    )
    return TrainingWorkflowConfig(
        SESSION=training_settings.SESSION,
        DATASET=training_settings.DATASET,
        DATAMODULE=training_settings.DATAMODULE,
        MODEL=training_settings.MODEL,
        TRAINING=training,
    )


def test_standard_staged_wrapper_uses_manual_controller(
    training_settings: TrainingWorkflowConfig,
) -> None:
    """The standard wrapper remains the manual-optimization control case."""
    settings = _build_standard_staged_settings(training_settings)
    components = BuildFactory().build_components(settings)
    model = components.model

    assert isinstance(model._optimization_controller, ManualOptimizationController)
    assert not model.automatic_optimization


def test_standard_staged_training_succeeds_with_lr_tuning(
    training_settings: TrainingWorkflowConfig,
) -> None:
    """Executor-level staged LR tuning should update stage 0 and keep stage 1 unchanged."""
    settings = _build_standard_staged_settings(training_settings)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api_train(settings)

    assert isinstance(result, TrainingResult)
    assert result.model_state is not None
    controller = cast(Any, result.model_state.model)._optimization_controller
    stage_0_lr = controller._program.stages[0].optimizer.param_groups[0]["lr"]
    stage_1_lr = controller._program.stages[1].optimizer.param_groups[0]["lr"]
    assert stage_0_lr != pytest.approx(1e-3)
    assert stage_1_lr == pytest.approx(1e-3)
    assert all(
        "Detected call of `lr_scheduler.step()` before `optimizer.step()`" not in str(w.message)
        for w in caught
    )


def test_executor_restores_manual_controller_after_lr_tuning(
    training_settings: TrainingWorkflowConfig,
) -> None:
    """VanillaExecutor must restore the staged manual controller after LR tuning.

    The executor projects the staged policy to a single-stage tuning policy,
    runs LR finding with a real controller swap, then restores the original
    controller before training.
    """
    from unittest.mock import patch

    from dlkit.engine.training.vanilla_executor import VanillaExecutor

    settings = _build_standard_staged_settings(training_settings)
    components = BuildFactory().build_components(settings)
    model = components.model

    assert isinstance(model._optimization_controller, ManualOptimizationController)
    original_controller = model._optimization_controller

    executor = VanillaExecutor()
    with patch.object(LRTuner, "tune", return_value=0.01):
        executor._apply_lr_tuning(model, components.datamodule, settings)

    assert model._optimization_controller is original_controller
    assert not model.automatic_optimization
    assert model.lr == pytest.approx(0.01)
