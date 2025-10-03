"""Strategy factory for composable execution following OCP."""

from __future__ import annotations

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

    def create_executor(self, settings: GeneralSettings) -> ITrainingExecutor:
        """Create composed execution strategy from settings.

        Args:
            settings: Configuration settings with feature flags

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

        # Always create a tracker (null object pattern eliminates conditionals)
        if self._is_mlflow_enabled(settings):
            # Import lazily from the concrete module so tests can patch it reliably
            from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker

            tracker = MLflowTracker(
                disable_autostart=False,
                skip_health_checks=False,
            )
        else:
            # Use null tracker when MLflow is disabled
            from dlkit.runtime.workflows.strategies.tracking import NullTracker

            tracker = NullTracker()

        # Always apply tracking layer (real or null)
        executor = TrackingDecorator(executor, tracker)

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
            if self._is_mlflow_enabled(settings):
                # Import and create the tracking adapter that wraps an existing MLflow tracker
                from dlkit.runtime.workflows.optimization.infrastructure.tracking import MLflowTrackingAdapter
                from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker

                # Create MLflow tracker for the optimization workflow with context management
                mlflow_tracker = MLflowTracker(
                    disable_autostart=False,
                    skip_health_checks=False,
                )

                # Get MLflow config
                mlflow_config = getattr(settings, "MLFLOW", None)

                # Determine experiment name using standard naming convention
                from dlkit.runtime.workflows.strategies.tracking import determine_experiment_name
                experiment_name = determine_experiment_name(settings, mlflow_config)

                session = getattr(settings, "SESSION", None)
                root_dir = getattr(session, "root_dir", None) if session is not None else None

                # Wrap it in our optimization tracking adapter with settings
                experiment_tracker = MLflowTrackingAdapter(
                    mlflow_tracker=mlflow_tracker,
                    mlflow_settings=mlflow_config,
                    session_name=experiment_name,
                    root_dir=root_dir,
                )

            # Use the clean architecture that fixes SOLID violations
            from dlkit.runtime.workflows.optimization import OptimizationServiceFactory

            factory = OptimizationServiceFactory(experiment_tracker=experiment_tracker)
            return factory.create_optimization_strategy(settings)
        else:
            # No compatibility layer - optimization must be explicitly enabled
            from dlkit.interfaces.api.domain import WorkflowError

            raise WorkflowError(
                "Optimization strategy requested but OPTUNA is not enabled. "
                "Enable OPTUNA in settings or use training strategy instead.",
                {"stage": "strategy_creation", "optuna_enabled": False},
            )

    def _is_mlflow_enabled(self, settings: GeneralSettings) -> bool:
        """Check if MLflow tracking is enabled in settings.

        Do not silently disable when package is missing — errors surface later.
        """
        mlflow_config = getattr(settings, "MLFLOW", None)
        return bool(mlflow_config and getattr(mlflow_config, "enabled", False))

    def _is_optuna_enabled(self, settings: GeneralSettings) -> bool:
        """Check if Optuna optimization is enabled in settings."""
        optuna_config = getattr(settings, "OPTUNA", None)
        return bool(optuna_config and getattr(optuna_config, "enabled", False))
