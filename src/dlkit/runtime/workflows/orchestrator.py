"""Workflow Orchestrator for training execution (Phase 1).

Composes: prepare (ops) -> build (factory) -> execute (strategy) -> finalize.
"""

from __future__ import annotations


from dlkit.interfaces.api.domain import TrainingResult, OptimizationResult
from dlkit.interfaces.api.tracking_hooks import TrackingHooks
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import (
    TrainingWorkflowConfig,
    InferenceWorkflowConfig,
    OptimizationWorkflowConfig,
)
from dlkit.tools.utils.logging_config import get_logger
from dlkit.tools.utils.error_handling import raise_error

from .factories.build_factory import BuildFactory
from .strategies.factory import ExecutionStrategyFactory

logger = get_logger(__name__, "orchestrator")


class ExecutionSelector:
    """Pure SOLID execution strategy selector using factory composition."""

    def __init__(self, factory: ExecutionStrategyFactory | None = None):
        self._factory = factory or ExecutionStrategyFactory()

    def select(
        self,
        settings: GeneralSettings
        | TrainingWorkflowConfig
        | InferenceWorkflowConfig
        | OptimizationWorkflowConfig,
        explicit: str | None = None,
        hooks: TrackingHooks | None = None,
    ):
        """Create execution strategy using SOLID factory composition."""
        # Log what features are detected
        features = []
        if settings.OPTUNA and getattr(settings.OPTUNA, "enabled", False):
            features.append("Optuna optimization")
        if settings.MLFLOW:
            features.append("MLflow tracking")
        if not features:
            features.append("vanilla training")

        feature_str = " + ".join(features)
        logger.info("Creating executor with {}", feature_str)

        # Use factory to create composed executor based on settings
        return self._factory.create_executor(settings, hooks=hooks)  # type: ignore[arg-type]

    def select_optimization(
        self,
        settings: GeneralSettings
        | TrainingWorkflowConfig
        | InferenceWorkflowConfig
        | OptimizationWorkflowConfig,
    ):
        """Create optimization strategy using SOLID factory composition."""
        # Log what features are detected
        features = []
        if settings.OPTUNA and getattr(settings.OPTUNA, "enabled", False):
            features.append("Optuna optimization")
        if settings.MLFLOW:
            features.append("MLflow tracking")
        if not features:
            features.append("vanilla training (adapted)")

        feature_str = " + ".join(features)
        logger.info("Creating optimization strategy with {}", feature_str)

        # Use factory to create composed optimization strategy based on settings
        return self._factory.create_optimization_strategy(settings)


class Orchestrator:
    """High-level training orchestrator."""

    def __init__(
        self, build_factory: BuildFactory | None = None, selector: ExecutionSelector | None = None
    ) -> None:
        self._build_factory = build_factory or BuildFactory()
        self._selector = selector or ExecutionSelector()

    def execute_training(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        hooks: TrackingHooks | None = None,
    ) -> TrainingResult:
        logger.info("Starting training workflow orchestration")
        try:
            # Suspend training (and MLflow/Optuna) when in inference mode
            if settings.SESSION and getattr(settings.SESSION, "inference", False):
                logger.warning("Training requested but inference mode is active")
                raise_error(
                    "Inference mode active: training, MLflow and Optuna are suspended. Use inference service instead."
                )
                # This line is never reached but helps type checker understand the flow
                return  # type: ignore[unreachable]

            # Build runtime components
            logger.debug("Building runtime components")
            components = self._build_factory.build_components(settings)
            logger.debug("Runtime components built successfully")

            # Select and run execution strategy
            exec_strategy = self._selector.select(settings, hooks=hooks)
            logger.info("Starting training execution with selected strategy")
            result = exec_strategy.execute(components, settings)
            logger.info("Training execution completed successfully")
            return result

        except Exception as e:
            raise_error("Training orchestration failed", e)

    def execute_optimization(
        self,
        settings: GeneralSettings | OptimizationWorkflowConfig,
    ) -> OptimizationResult:
        """Execute optimization workflow using factory pattern for proper composition.

        This now uses the ExecutionSelector factory to create the appropriate
        optimization strategy, ensuring proper MLflow integration when both
        Optuna and MLflow are enabled (parent study run + nested trial runs).
        """
        logger.info("Starting optimization workflow orchestration")
        try:
            # Optimization should not run in inference mode
            if settings.SESSION and getattr(settings.SESSION, "inference", False):
                logger.warning("Optimization requested but inference mode is active")
                raise_error("Inference mode active: optimization suspended.")
                # This line is never reached but helps type checker understand the flow
                return  # type: ignore[unreachable]

            # Use factory pattern for proper strategy composition
            optimization_strategy = self._selector.select_optimization(settings)
            logger.info("Starting optimization execution with selected strategy")
            result = optimization_strategy.execute_optimization(settings)
            logger.info("Optimization execution completed successfully")
            return result

        except Exception as e:
            raise_error("Optimization orchestration failed", e)


# Execution classes moved to strategies/execution.py to avoid bloat here
