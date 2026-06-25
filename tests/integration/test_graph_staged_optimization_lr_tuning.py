"""Integration tests for graph staged optimization and LR tuning behavior."""

from __future__ import annotations

import warnings

from dlkit.common import TrainingResult
from dlkit.engine.training.optimization.controllers import ManualOptimizationController
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import TrainingJobConfig
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
from dlkit.interfaces.api import train as api_train


def _build_staged_graph_settings(
    graph_settings: TrainingJobConfig,
    *,
    enable_lr_tuning: bool,
) -> TrainingJobConfig:
    """Create a graph workflow with sequential staged optimizers.

    Args:
        graph_settings: Base graph training configuration.
        enable_lr_tuning: Whether to include LR tuning configuration.

    Returns:
        Modified TrainingJobConfig with staged optimizer policy.
    """
    assert graph_settings.training is not None
    lr_tuner = (
        LRTunerSettings(min_lr=1e-6, max_lr=0.1, num_training=2) if enable_lr_tuning else None
    )
    updated_training = graph_settings.training.model_copy(
        update={
            "lr_tuner": lr_tuner,
            "trainer": TrainerSettings.model_validate(
                {
                    "fast_dev_run": False,
                    "enable_checkpointing": False,
                    "accelerator": "cpu",
                    "enable_progress_bar": False,
                    "enable_model_summary": False,
                    "limit_train_batches": 2,
                    "max_epochs": 1,
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
    return graph_settings.model_copy(update={"training": updated_training})


def test_graph_staged_wrapper_uses_manual_controller(
    graph_settings: TrainingJobConfig,
) -> None:
    """Graph staged policies should build a manual controller before training starts."""
    settings = _build_staged_graph_settings(graph_settings, enable_lr_tuning=False)
    components = BuildFactory().build_components(settings)
    model = components.model

    assert isinstance(model._optimization_controller, ManualOptimizationController)
    assert not model.automatic_optimization


def test_graph_staged_training_succeeds_without_lr_tuning(
    graph_settings: TrainingJobConfig,
) -> None:
    """Sequential graph staged training should complete successfully."""
    settings = _build_staged_graph_settings(graph_settings, enable_lr_tuning=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api_train(settings)

    assert isinstance(result, TrainingResult)
    assert all(
        "Detected call of `lr_scheduler.step()` before `optimizer.step()`" not in str(w.message)
        for w in caught
    )


def test_graph_staged_training_succeeds_with_lr_tuning(
    graph_settings: TrainingJobConfig,
) -> None:
    """Sequential graph staged training should complete even when LR tuning is enabled."""
    settings = _build_staged_graph_settings(graph_settings, enable_lr_tuning=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = api_train(settings)

    assert isinstance(result, TrainingResult)
    assert all(
        "Detected call of `lr_scheduler.step()` before `optimizer.step()`" not in str(w.message)
        for w in caught
    )
