"""Runtime-owned unified execution entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)

from ._override_types import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from ._settings import coerce_general_settings
from .optimization import optimize
from .training import train


def execute(
    settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult:
    """Dispatch between runtime training and optimization entrypoints."""
    effective = coerce_general_settings(settings)
    session = getattr(effective, "SESSION", None)
    if session and getattr(session, "inference", False):
        raise WorkflowError(
            "execute() no longer supports inference workflows. Use dlkit.load_model() instead.",
            {"workflow": "inference"},
        )

    shared_overrides = dict(overrides or {})

    optuna_config = getattr(effective, "OPTUNA", None)
    if optuna_config and getattr(optuna_config, "enabled", False):
        optimization_overrides = cast(
            OptimizationOverrides,
            {
                key: value
                for key, value in shared_overrides.items()
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
        return optimize(
            effective,
            optimization_overrides,
        )

    training_overrides = cast(
        TrainingOverrides,
        {
            key: value
            for key, value in shared_overrides.items()
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
    return train(
        effective,
        training_overrides,
        hooks=hooks,
    )
