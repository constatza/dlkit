"""Runtime-owned unified execution entrypoint."""

from __future__ import annotations

from dlkit.common import OptimizationResult, TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import ConvergenceResult
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from ._override_types import (
    ConvergenceOverrides,
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
    require_override_model,
)
from .convergence import converge
from .optimization import optimize
from .training import train


def execute(
    settings: TrainingJobConfig
    | SearchJobConfig
    | InferenceJobConfig
    | ConvergenceJobConfig
    | JobConfig,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> TrainingResult | OptimizationResult | ConvergenceResult:
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
            return optimize(
                settings, optimization_overrides if override_payload else None, hooks=hooks
            )

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

        case ConvergenceJobConfig():
            convergence_overrides = ConvergenceOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key
                    in {
                        "experiment_name",
                        "run_name",
                        "tags",
                        "sizes",
                        "repeats",
                        "target",
                    }
                }
            )
            return converge(settings, convergence_overrides if override_payload else None)

        case _:
            raise WorkflowError(
                f"Unsupported workflow settings type: {type(settings).__name__}",
                {"workflow": "unknown"},
            )
