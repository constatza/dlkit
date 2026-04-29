"""Runtime-owned optimization workflow entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import OptimizationResult
from dlkit.common.errors import WorkflowError
from dlkit.infrastructure.config.workflow_configs import OptimizationWorkflowConfig

from ..optimization.factory import OptimizationServiceFactory
from ._entrypoint_context import EntrypointContext
from ._override_types import OptimizationOverrides, require_override_model


def optimize(
    settings: OptimizationWorkflowConfig,
    overrides: OptimizationOverrides | None = None,
) -> OptimizationResult:
    """Run hyperparameter optimization through runtime orchestration."""
    validated_overrides = require_override_model(overrides, OptimizationOverrides)
    context = EntrypointContext.prepare(
        settings,
        validated_overrides,
        workflow_name="optimization",
    )

    try:
        opt_settings = cast(OptimizationWorkflowConfig, context.settings)
        base_factory = OptimizationServiceFactory()
        experiment_tracker = base_factory.create_experiment_tracker(opt_settings)
        strategy_factory = OptimizationServiceFactory(experiment_tracker=experiment_tracker)
        optimization_strategy = strategy_factory.create_optimization_strategy(opt_settings)

        def run() -> OptimizationResult:
            return optimization_strategy.execute_optimization(opt_settings)

        def run_with_tracker() -> OptimizationResult:
            if experiment_tracker is None:
                return run()
            with experiment_tracker:
                return run()

        result = context.run_with_path_context(run_with_tracker)

        return OptimizationResult(
            best_trial=result.best_trial,
            training_result=result.training_result,
            study_summary=result.study_summary,
            duration_seconds=context.elapsed(),
        )
    except Exception as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError(
            f"Optimization execution failed: {exc!s}",
            {"workflow": "optimization", "error": str(exc)},
        ) from exc
