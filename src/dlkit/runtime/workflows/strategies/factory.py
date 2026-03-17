"""Strategy factory for composable execution following OCP."""

from __future__ import annotations

from dlkit.interfaces.api.tracking_hooks import TrackingHooks
from dlkit.tools.config import GeneralSettings

from .core import ITrainingExecutor, VanillaExecutor
from .core.interfaces import IOptimizationStrategy
from .tracking import TrackingDecorator


class ExecutionStrategyFactory:
    """Factory for creating composable execution strategies following OCP.

    Enables runtime composition of:
    - Core execution (vanilla)
    - Optional tracking (MLflow)
    - Optional optimization (Optuna)

    All combinations are supported orthogonally.
    Supports both training executors and optimization strategies.
    """

    def create_executor(
        self,
        settings: GeneralSettings,
        hooks: TrackingHooks | None = None,
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
            from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker
            from dlkit.runtime.workflows.strategies.tracking.uri_resolver import local_host_alive

            tracker = MLflowTracker(
                disable_autostart=False,
                probe=local_host_alive,
            )
        else:
            from dlkit.runtime.workflows.strategies.tracking import NullTracker

            tracker = NullTracker()

        # Always apply tracking layer (real or null), forwarding hooks
        executor = TrackingDecorator(executor, tracker, hooks=hooks)

        # Note: Optimization layer is NOT added here - that's handled by create_optimization_strategy
        # This method returns: TrackingDecorator -> VanillaExecutor
        return executor

    def create_optimization_strategy(self, settings: GeneralSettings) -> IOptimizationStrategy:
        """Create optimization strategy from settings.

        This method uses the clean domain-driven architecture that properly
        models the Study → Trial → Best Retrain hierarchy with nested MLflow runs.

        Args:
            settings: Configuration settings with feature flags

        Returns:
            Clean optimization strategy with proper SOLID architecture

        Raises:
            WorkflowError: If optimization is requested but not properly configured
        """
        if self._is_optuna_enabled(settings):
            # Create experiment tracker here to avoid duplication
            experiment_tracker = None
            if self._has_mlflow_config(settings):
                from dlkit.runtime.workflows.optimization.infrastructure.tracking import (
                    MLflowTrackingAdapter,
                )
                from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker
                from dlkit.runtime.workflows.strategies.tracking.uri_resolver import local_host_alive

                mlflow_tracker = MLflowTracker(
                    disable_autostart=False,
                    probe=local_host_alive,
                )

                mlflow_config = getattr(settings, "MLFLOW", None)

                from dlkit.runtime.workflows.strategies.tracking import determine_experiment_name

                experiment_name = determine_experiment_name(settings, mlflow_config)

                session = getattr(settings, "SESSION", None)
                root_dir = getattr(session, "root_dir", None) if session is not None else None

                experiment_tracker = MLflowTrackingAdapter(
                    mlflow_tracker=mlflow_tracker,
                    mlflow_settings=mlflow_config,
                    session_name=experiment_name,
                    root_dir=root_dir,
                )

            from dlkit.runtime.workflows.optimization import OptimizationServiceFactory

            factory = OptimizationServiceFactory(experiment_tracker=experiment_tracker)
            return factory.create_optimization_strategy(settings)
        else:
            from dlkit.interfaces.api.domain import WorkflowError

            raise WorkflowError(
                "Optimization strategy requested but OPTUNA is not enabled. "
                "Enable OPTUNA in settings or use training strategy instead.",
                {"stage": "strategy_creation", "optuna_enabled": False},
            )

    def _has_mlflow_config(self, settings: GeneralSettings) -> bool:
        """Check if MLflow configuration section is present in settings.

        MLflow is enabled whenever the [MLFLOW] section exists — no separate
        ``enabled`` flag is needed.
        """
        return bool(getattr(settings, "MLFLOW", None))

    def _has_mlflow_config_or_env(self, settings: GeneralSettings) -> bool:
        """Check if MLflow should be activated for training.

        Activates when the ``[MLFLOW]`` config section exists, when the
        standard ``MLFLOW_TRACKING_URI`` environment variable is set, or when
        a local MLflow server is detected at the default address.  The local
        probe only runs when neither config nor env var is present (short-circuit).
        """
        import os

        from dlkit.runtime.workflows.strategies.tracking.uri_resolver import local_host_alive

        env_uri = os.getenv("MLFLOW_TRACKING_URI")
        has_user_http_uri = bool(
            env_uri and (env_uri.startswith("http://") or env_uri.startswith("https://"))
        )
        return self._has_mlflow_config(settings) or has_user_http_uri or local_host_alive()

    def _is_optuna_enabled(self, settings: GeneralSettings) -> bool:
        """Check if Optuna optimization is enabled in settings."""
        optuna_config = getattr(settings, "OPTUNA", None)
        return bool(optuna_config and getattr(optuna_config, "enabled", False))
