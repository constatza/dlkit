"""Runtime-owned unified execution entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.infrastructure.config.workflow_configs import (
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.config.workflow_types import WorkflowConfig

from ._override_types import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from .optimization import optimize
from .training import train


def execute(
    settings: WorkflowConfig,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult:
    """Dispatch between runtime training and optimization entrypoints."""
    match settings:
        case OptimizationWorkflowConfig():
            optimization_overrides = cast(
                OptimizationOverrides,
                {
                    key: value
                    for key, value in (overrides or {}).items()
                    if key
                    in {
                        "checkpoint_path",
                        "root_dir",
                        "output_dir",
                        "data_dir",
                        "trials",
                        "study_name",
                        "experiment_name",
                        "run_name",
                        "enable_optuna",
                        "register_model",
                        "tags",
                    }
                },
            )
            return optimize(settings, optimization_overrides)

        case InferenceWorkflowConfig():
            raise WorkflowError(
                "execute() does not support inference workflows. Use dlkit.load_model() instead.",
                {"workflow": "inference"},
            )

        case TrainingWorkflowConfig():
            training_overrides = cast(
                TrainingOverrides,
                {
                    key: value
                    for key, value in (overrides or {}).items()
                    if key
                    in {
                        "checkpoint_path",
                        "root_dir",
                        "output_dir",
                        "data_dir",
                        "epochs",
                        "batch_size",
                        "learning_rate",
                        "experiment_name",
                        "run_name",
                        "register_model",
                        "tags",
                        "loss_function",
                        "loss_module",
                    }
                },
            )
            return train(settings, training_overrides, hooks=hooks)
