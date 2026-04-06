"""Runtime-owned optimization workflow entrypoint."""

from __future__ import annotations

from dlkit.common import OptimizationResult
from dlkit.common.errors import WorkflowError

from ..optimization.factory import OptimizationServiceFactory
from ._entrypoint_context import EntrypointContext
from ._override_types import OptimizationOverrides
from ._settings import WorkflowSettings


def optimize(
    settings: WorkflowSettings,
    overrides: OptimizationOverrides | None = None,
) -> OptimizationResult:
    """Run hyperparameter optimization through runtime orchestration."""
    context = EntrypointContext.prepare(settings, overrides, workflow_name="optimization")

    try:
        base_factory = OptimizationServiceFactory()
        experiment_tracker = base_factory.create_experiment_tracker(context.settings)
        strategy_factory = OptimizationServiceFactory(experiment_tracker=experiment_tracker)
        optimization_strategy = strategy_factory.create_optimization_strategy(context.settings)

        def run() -> OptimizationResult:
            return optimization_strategy.execute_optimization(context.settings)

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
