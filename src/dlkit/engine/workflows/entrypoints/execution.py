"""Runtime-owned unified execution entrypoint."""

from __future__ import annotations

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from ._override_types import (
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
    require_override_model,
)
from .optimization import optimize
from .training import train


def execute(
    settings: TrainingJobConfig | SearchJobConfig | InferenceJobConfig | JobConfig,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult:
    """Dispatch between runtime training and optimization entrypoints."""
    validated_overrides = require_override_model(overrides, ExecutionOverrides)
    override_payload = (
        validated_overrides.model_dump(exclude_none=True) if validated_overrides is not None else {}
    )

    match settings:
        case SearchJobConfig():
            optimization_overrides = OptimizationOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key
                    in {
                        "checkpoint_path",
                        "trials",
                        "study_name",
                        "experiment_name",
                        "run_name",
                        "enable_optuna",
                        "register_model",
                        "tags",
                    }
                }
            )
            return optimize(settings, optimization_overrides if override_payload else None)

        case InferenceJobConfig():
            raise WorkflowError(
                "execute() does not support inference workflows. Use dlkit.load_model() instead.",
                {"workflow": "inference"},
            )

        case TrainingJobConfig():
            training_overrides = TrainingOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key
                    in {
                        "checkpoint_path",
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
                }
            )
            return train(settings, training_overrides if override_payload else None, hooks=hooks)

        case _:
            raise WorkflowError(
                f"Unsupported workflow settings type: {type(settings).__name__}",
                {"workflow": "unknown"},
            )
