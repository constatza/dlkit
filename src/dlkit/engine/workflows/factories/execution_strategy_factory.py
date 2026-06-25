"""Strategy factory for composable execution following OCP."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking import uri_resolver
from dlkit.engine.tracking.interfaces import NullTracker
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training import ITrainingExecutor, VanillaExecutor
from dlkit.infrastructure.config.job_config import JobConfig

if TYPE_CHECKING:
    from dlkit.engine.workflows.factories.build_strategy import WorkflowSettings


def _default_probe() -> bool:
    """Use the tracking resolver at call time so tests can monkeypatch it."""
    return uri_resolver.local_host_alive()


class ExecutionStrategyFactory:
    """Factory for creating composable execution strategies following OCP.

    Enables runtime composition of:
    - Core execution (vanilla)
    - Optional tracking (MLflow)
    """

    def __init__(self, probe: Callable[[], bool] = _default_probe) -> None:
        self._probe = probe

    def create_executor(
        self,
        settings: JobConfig,
        hooks: LifecycleHooks | None = None,
    ) -> ITrainingExecutor:
        """Create composed execution strategy from settings.

        Args:
            settings: Configuration settings with feature flags
            hooks: Optional functional extension points for tracking lifecycle

        Returns:
            Composed training executor with requested capabilities

        Examples:
            - Vanilla only: VanillaExecutor
            - Vanilla + MLflow: TrackingDecorator(VanillaExecutor, MLflowTracker)
            - Vanilla + Optuna: OptimizationDecorator(VanillaExecutor, OptunaOptimizer)
            - All features: OptimizationDecorator(TrackingDecorator(...), OptunaOptimizer, MLflowTracker)
        """
        # Start with core vanilla executor
        executor: ITrainingExecutor = VanillaExecutor()

        # Use real tracker when MLFLOW section is present or MLFLOW_TRACKING_URI env var is set
        if self._has_mlflow_config_or_env(settings):
            tracker = MLflowTracker(
                disable_autostart=False,
                probe=self._probe,
            )
        else:
            tracker = NullTracker()

        # Always apply tracking layer (real or null), forwarding hooks
        executor = TrackingDecorator(executor, tracker, hooks=hooks)

        # This method returns TrackingDecorator -> VanillaExecutor.
        return executor

    def _has_mlflow_config_or_env(
        self,
        settings: JobConfig,
    ) -> bool:
        """Check if MLflow should be activated for training.

        Activates when tracking.backend == "mlflow", when MLFLOW_TRACKING_URI
        env var is an HTTP URI, or when a local MLflow server is detected.
        """
        import os

        if settings.tracking.backend == "mlflow":
            return True
        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        has_user_http_uri = bool(
            env_uri and (env_uri.startswith("http://") or env_uri.startswith("https://"))
        )
        return has_user_http_uri or self._probe()

    def _is_optimization_workflow(
        self,
        settings: WorkflowSettings,
    ) -> bool:
        """Check if this is an optimization workflow based on config type."""
        from dlkit.infrastructure.config.job_config import SearchJobConfig

        return isinstance(settings, SearchJobConfig)


def create_execution_strategy(
    settings: WorkflowSettings,
    hooks: LifecycleHooks | None = None,
) -> ITrainingExecutor:
    """Convenience function to create a composed execution strategy.

    Args:
        settings: Configuration settings with feature flags
        hooks: Optional functional extension points for tracking lifecycle

    Returns:
        Composed training executor with requested capabilities
    """
    return ExecutionStrategyFactory().create_executor(settings, hooks=hooks)
