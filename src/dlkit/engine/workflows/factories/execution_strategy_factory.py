"""Strategy factory for composable execution following OCP."""

from __future__ import annotations

from collections.abc import Callable

from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking import uri_resolver
from dlkit.engine.tracking.interfaces import NullTracker
from dlkit.engine.tracking.mlflow_tracker import MLflowTracker
from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
from dlkit.engine.training import ITrainingExecutor, VanillaExecutor
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)


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
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
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

    def _has_mlflow_config(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> bool:
        """Check if MLflow configuration section is present in settings.

        MLflow is enabled whenever the [MLFLOW] section exists — no separate
        ``enabled`` flag is needed.
        """
        return bool(getattr(settings, "MLFLOW", None))

    def _has_mlflow_config_or_env(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> bool:
        """Check if MLflow should be activated for training.

        Activates when the ``[MLFLOW]`` config section exists, when the
        standard ``MLFLOW_TRACKING_URI`` environment variable is set, or when
        a local MLflow server is detected at the default address.  The local
        probe only runs when neither config nor env var is present (short-circuit).
        """
        import os

        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        has_user_http_uri = bool(
            env_uri and (env_uri.startswith("http://") or env_uri.startswith("https://"))
        )
        return self._has_mlflow_config(settings) or has_user_http_uri or self._probe()

    def _is_optuna_enabled(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> bool:
        """Check if Optuna optimization is enabled in settings."""
        optuna_config = getattr(settings, "OPTUNA", None)
        return bool(optuna_config and getattr(optuna_config, "enabled", False))


def create_execution_strategy(
    settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
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
