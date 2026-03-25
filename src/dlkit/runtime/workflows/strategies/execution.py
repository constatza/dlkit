"""Pure SOLID execution strategies - no backward compatibility."""

# Re-export the new SOLID interfaces and implementations
from .core import ITrainingExecutor, VanillaExecutor
from .factory import ExecutionStrategyFactory, create_execution_strategy

# OptimizationDecorator removed - use clean architecture instead
from .optuna.optuna_optimizer import OptunaOptimizer
from .tracking import MLflowTracker, TrackingDecorator

# For direct usage - no backward compatibility wrappers
__all__ = [
    "ExecutionStrategyFactory",
    "ITrainingExecutor",
    "MLflowTracker",
    "OptunaOptimizer",
    "TrackingDecorator",
    "VanillaExecutor",
    "create_execution_strategy",
]
