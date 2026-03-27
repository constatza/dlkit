"""DLKit API domain layer.

This module contains the core domain models, protocols, and errors
that define the business logic of DLKit without framework dependencies.
"""

from dlkit.domain import InferenceResult, ModelState, OptimizationResult, TrainingResult

from .errors import (
    ConfigurationError,
    DLKitError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)
from .override_types import BasicOverrides, MLflowOverrides, PathOverrides, TrainingOverrides
from .precision import (
    PrecisionContext,
    PrecisionProvider,
    get_precision_context,
    precision_override,
)
from .protocols import ExecutionStrategy, StrategyFactory, WorkflowOperation

__all__ = [
    # Models
    "ModelState",
    "TrainingResult",
    "InferenceResult",
    "OptimizationResult",
    # Override types
    "BasicOverrides",
    "MLflowOverrides",
    "PathOverrides",
    "TrainingOverrides",
    # Precision
    "PrecisionContext",
    "PrecisionProvider",
    "precision_override",
    "get_precision_context",
    # Protocols
    "ExecutionStrategy",
    "WorkflowOperation",
    "StrategyFactory",
    # Errors
    "DLKitError",
    "ConfigurationError",
    "WorkflowError",
    "StrategyError",
    "ModelStateError",
    "PluginError",
]
