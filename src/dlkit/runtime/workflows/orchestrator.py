"""Workflow Orchestrator for training execution (Phase 1).

Composes: prepare (ops) -> build (factory) -> execute (strategy) -> finalize.
"""

from __future__ import annotations

from dlkit.shared import TrainingResult
from dlkit.shared.hooks import LifecycleHooks
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.tools.utils.error_handling import raise_error
from dlkit.tools.utils.logging_config import get_logger

from .factories.build_factory import BuildFactory
from .factories.execution_strategy_factory import ExecutionStrategyFactory
from .optimization.factory import OptimizationServiceFactory

logger = get_logger(__name__, "orchestrator")


class WorkflowExecutionSelector:
    """Routes workflow settings to the appropriate execution strategy (training or optimization)."""

    def __init__(self, factory: ExecutionStrategyFactory | None = None):
        self._factory = factory or ExecutionStrategyFactory()

    def select(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        explicit: str | None = None,
        hooks: LifecycleHooks | None = None,
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
        return self._factory.create_executor(settings, hooks=hooks)

    def select_optimization(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ):
        """Create optimization strategy from the runtime workflows layer."""
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

        return OptimizationServiceFactory().create_optimization_strategy(settings)


class Orchestrator:
    """High-level training orchestrator."""

    def __init__(
        self,
        build_factory: BuildFactory | None = None,
        selector: WorkflowExecutionSelector | None = None,
    ) -> None:
        self._build_factory = build_factory or BuildFactory()
        self._selector = selector or WorkflowExecutionSelector()

    def execute_training(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        hooks: LifecycleHooks | None = None,
    ) -> TrainingResult:
        logger.info("Starting training workflow orchestration")
        try:
            # Suspend training (and MLflow/Optuna) when in inference mode
            if settings.SESSION and getattr(settings.SESSION, "inference", False):
                logger.warning("Training requested but inference mode is active")
                raise_error(
                    "Inference mode active: training, MLflow and Optuna are suspended. Use inference service instead."
                )

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


# Execution classes moved to strategies/execution.py to avoid bloat here
