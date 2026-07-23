"""Runtime-owned unified execution entrypoint."""

from __future__ import annotations

from collections.abc import Mapping

from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.results import WorkflowResult
from dlkit.infrastructure.config.job_config import (
    ConvergenceJobConfig,
    InferenceJobConfig,
    JobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)

from ._override_types import (
    ConvergenceOverrides,
    EvaluationOverrides,
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
    require_override_model,
)
from .convergence import converge
from .evaluate import evaluate
from .optimization import optimize
from .training import train

# Per-workflow allow-list of override keys accepted from the raw override
# payload before validating into the workflow-specific overrides model.
_ALLOWED_OVERRIDE_KEYS: Mapping[type[JobConfig], frozenset[str]] = {
    SearchJobConfig: frozenset(
        {
            "checkpoint_path",
            "trials",
            "study_name",
            "experiment_name",
            "run_name",
            "enable_optuna",
            "register_model",
            "tags",
        }
    ),
    TrainingJobConfig: frozenset(
        {
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
    ),
    ConvergenceJobConfig: frozenset(
        {
            "experiment_name",
            "run_name",
            "tags",
            "sizes",
            "repeats",
            "target",
        }
    ),
    InferenceJobConfig: frozenset(
        {
            "checkpoint_path",
            "experiment_name",
            "run_name",
            "tags",
            "batch_size",
            "split",
            "device",
        }
    ),
}


def execute(
    settings: TrainingJobConfig
    | SearchJobConfig
    | InferenceJobConfig
    | ConvergenceJobConfig
    | JobConfig,
    overrides: ExecutionOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> WorkflowResult:
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
                    if key in _ALLOWED_OVERRIDE_KEYS[SearchJobConfig]
                }
            )
            return optimize(
                settings, optimization_overrides if override_payload else None, hooks=hooks
            )

        case InferenceJobConfig():
            evaluation_overrides = EvaluationOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key in _ALLOWED_OVERRIDE_KEYS[InferenceJobConfig]
                }
            )
            return evaluate(
                settings, evaluation_overrides if override_payload else None, hooks=hooks
            )

        case TrainingJobConfig():
            training_overrides = TrainingOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key in _ALLOWED_OVERRIDE_KEYS[TrainingJobConfig]
                }
            )
            return train(settings, training_overrides if override_payload else None, hooks=hooks)

        case ConvergenceJobConfig():
            convergence_overrides = ConvergenceOverrides.model_validate(
                {
                    key: value
                    for key, value in override_payload.items()
                    if key in _ALLOWED_OVERRIDE_KEYS[ConvergenceJobConfig]
                }
            )
            return converge(settings, convergence_overrides if override_payload else None)

        case _:
            raise WorkflowError(
                f"Unsupported workflow settings type: {type(settings).__name__}",
                {"workflow": "unknown"},
            )
