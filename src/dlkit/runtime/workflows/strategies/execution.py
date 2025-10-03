"""Pure SOLID execution strategies - no backward compatibility."""

# Re-export the new SOLID interfaces and implementations
from .core import ITrainingExecutor, VanillaExecutor
from .tracking import TrackingDecorator, MLflowTracker

# OptimizationDecorator removed - use clean architecture instead
from .optuna.optuna_optimizer import OptunaOptimizer
from .factory import ExecutionStrategyFactory, create_execution_strategy

# For direct usage - no backward compatibility wrappers
__all__ = [
    "ITrainingExecutor",
    "VanillaExecutor",
    "TrackingDecorator",
    "MLflowTracker",
    "OptunaOptimizer",
    "ExecutionStrategyFactory",
    "create_execution_strategy",
]
