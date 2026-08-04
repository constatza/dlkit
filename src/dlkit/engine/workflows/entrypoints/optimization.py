"""Runtime-owned optimization workflow entrypoint."""

from __future__ import annotations

from typing import cast

from dlkit.common import OptimizationResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from ..optimization.factory import OptimizationServiceFactory
from ._entrypoint_context import EntrypointContext
from ._override_types import OptimizationOverrides, require_override_model

logger = get_logger(__name__)


def optimize(
    settings: SearchJobConfig,
    overrides: OptimizationOverrides | None = None,
    *,
    hooks: LifecycleHooks | None = None,
) -> OptimizationResult:
    """Run hyperparameter optimization through runtime orchestration."""
    logger.info(
        "Optimization | study={} n_trials={} direction={}",
        getattr(getattr(settings, "search", None), "study_name", None) or "<auto>",
        getattr(getattr(settings, "search", None), "n_trials", "?"),
        getattr(getattr(settings, "search", None), "direction", "?"),
    )
    validated_overrides = require_override_model(overrides, OptimizationOverrides)
    context = EntrypointContext.prepare(
        settings,
        validated_overrides,
        workflow_name="optimization",
    )

    def run_optimization() -> OptimizationResult:
        opt_settings = cast(SearchJobConfig, context.settings)
        base_factory = OptimizationServiceFactory()
        experiment_tracker = base_factory.create_experiment_tracker(opt_settings, hooks=hooks)
        strategy_factory = OptimizationServiceFactory(
            experiment_tracker=experiment_tracker, hooks=hooks
        )
        optimization_strategy = strategy_factory.create_optimization_strategy(opt_settings)

        if experiment_tracker is None:
            result = optimization_strategy.execute_optimization(opt_settings)
        else:
            with experiment_tracker:
                result = optimization_strategy.execute_optimization(opt_settings)

        return OptimizationResult(
            best_trial=result.best_trial,
            training_result=result.training_result,
            study_summary=result.study_summary,
            duration_seconds=context.elapsed(),
            mlflow_run_id=result.mlflow_run_id,
            mlflow_tracking_uri=result.mlflow_tracking_uri,
        )

    return context.run(run_optimization, error_message="Optimization execution failed")
