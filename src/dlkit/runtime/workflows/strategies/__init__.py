"""Pure SOLID-compliant execution and optimization strategies for DLKit workflows."""

# Core SOLID architecture exports
from .core import ITrainingExecutor, VanillaExecutor
from .core.interfaces import IOptimizationStrategy
from .factory import ExecutionStrategyFactory
from .optuna.interfaces import IHyperparameterOptimizer
from .optuna.optuna_optimizer import OptunaOptimizer
from .tracking import IExperimentTracker, MLflowTracker, TrackingDecorator

__all__ = [
    # Core SOLID architecture
    "ITrainingExecutor",
    "VanillaExecutor",
    "IExperimentTracker",
    "MLflowTracker",
    "TrackingDecorator",
    "IHyperparameterOptimizer",
    "OptunaOptimizer",
    "ExecutionStrategyFactory",
    # Optimization interfaces
    "IOptimizationStrategy",
]
