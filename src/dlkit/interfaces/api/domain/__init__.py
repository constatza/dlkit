"""DLKit API domain layer.

This module contains the core domain models, protocols, and errors
that define the business logic of DLKit without framework dependencies.
"""

from dlkit.shared import (
    ConfigurationError,
    DLKitError,
    InferenceResult,
    ModelState,
    ModelStateError,
    OptimizationResult,
    PluginError,
    StrategyError,
    TrainingResult,
    WorkflowError,
)
from dlkit.tools.precision import (
    PrecisionContext,
    PrecisionProvider,
    get_precision_context,
    precision_override,
)

from .override_types import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from .protocols import ExecutionStrategy, StrategyFactory, WorkflowOperation

__all__ = [
    # Models
    "ModelState",
    "TrainingResult",
    "InferenceResult",
    "OptimizationResult",
    # Override types
    "ExecutionOverrides",
    "OptimizationOverrides",
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
