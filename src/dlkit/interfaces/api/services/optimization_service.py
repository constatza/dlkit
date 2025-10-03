"""Optimization service using the new Orchestrator (no legacy context)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dlkit.interfaces.api.domain import OptimizationResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.runtime.workflows.orchestrator import Orchestrator
from dlkit.interfaces.api.overrides.path_context import (
    path_override_context,
    get_current_path_context,
)

if TYPE_CHECKING:
    from dlkit.runtime.workflows.optimization.strategy import OptimizationStrategy


class OptimizationService:
    """Service for executing hyperparameter optimization workflows.

    Uses Orchestrator optimization path to run Optuna and return an
    OptimizationResult. Tracker context management now lives inside the
    orchestrator so this service can focus on request-level concerns.
    """

    def __init__(self) -> None:
        """Initialize optimization service."""
        self.service_name = "optimization_service"

    def execute_optimization(
        self,
        settings: GeneralSettings,
        trials: int = 100,
        checkpoint_path: Path | None = None,
    ) -> OptimizationResult:
        """Execute hyperparameter optimization workflow.

        This service owns the experiment tracker context lifecycle. The tracker
        context is entered here before optimization work begins and exited after
        completion, ensuring proper resource management at the service boundary.

        Args:
            settings: DLKit configuration settings
            trials: Number of optimization trials to run
            checkpoint_path: Optional checkpoint path for resuming

        Returns:
            OptimizationResult on success; raises WorkflowError on failure
        """
        start_time = time.time()

        try:
            # Apply settings-defined root_dir if present and not already overridden
            overrides: dict[str, Any] = {}
            ctx = get_current_path_context()
            try:
                root_from_cfg = getattr(getattr(settings, "SESSION", None), "root_dir", None)
                if root_from_cfg and not (ctx and getattr(ctx, "root_dir", None)):
                    overrides["root_dir"] = root_from_cfg
            except Exception:
                pass

            # Create orchestrator and get the optimization strategy
            orch = Orchestrator()
            optimization_strategy = orch._selector.select_optimization(settings)

            # Import at runtime to avoid circular dependency with the strategy module
            from dlkit.runtime.workflows.optimization.strategy import OptimizationStrategy as CleanOptimizationStrategy
            from dlkit.runtime.workflows.optimization.factory import OptimizationServiceFactory

            # Get experiment tracker for service-level context management
            experiment_tracker = None
            if isinstance(optimization_strategy, CleanOptimizationStrategy):
                factory = OptimizationServiceFactory()
                experiment_tracker = factory._create_experiment_tracker(settings)

            def _execute_clean_strategy() -> OptimizationResult:
                return optimization_strategy.execute_optimization(settings)

            def _execute_fallback() -> OptimizationResult:
                return orch.execute_optimization(settings)

            runner = _execute_fallback
            if isinstance(optimization_strategy, CleanOptimizationStrategy):
                runner = _execute_clean_strategy

            def _run_with_tracker() -> OptimizationResult:
                if experiment_tracker is not None:
                    with experiment_tracker:
                        return runner()
                return runner()

            if overrides:
                with path_override_context(overrides):
                    opt_res = _run_with_tracker()
            else:
                opt_res = _run_with_tracker()

            # Add duration
            duration = time.time() - start_time
            final_result = OptimizationResult(
                best_trial=opt_res.best_trial,
                training_result=opt_res.training_result,
                study_summary=opt_res.study_summary,
                duration_seconds=duration,
            )
            return final_result

        except Exception as e:
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Optimization execution failed: {str(e)}",
                {"service": self.service_name, "error": str(e)},
            )

    def get_optimization_progress(self, study_name: str) -> dict[str, Any]:
        """Get progress information for an ongoing optimization study.

        Args:
            study_name: Name of the Optuna study

        Returns:
            Dictionary containing progress information
        """
        try:
            # This would interface with Optuna's study storage
            # Implementation would depend on your Optuna setup
            progress_info = {
                "study_name": study_name,
                "status": "running",
                "completed_trials": 0,
                "best_value": None,
                "best_params": None,
            }
            return progress_info
        except Exception as e:
            raise WorkflowError(
                f"Failed to get optimization progress: {str(e)}",
                {"service": self.service_name, "study_name": study_name},
            )
