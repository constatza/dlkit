"""DLKit API domain layer.

This module contains the core domain models, protocols, and errors
that define the business logic of DLKit without framework dependencies.
"""

from dlkit.common import (
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
from dlkit.infrastructure.precision import (
    PrecisionContext,
    PrecisionProvider,
    get_precision_context,
    precision_override,
)

from .override_types import (
    ExecutionOverrides,
    OptimizationOverrides,
    RuntimeOverrideModel,
    TrainingOverrides,
)
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
    "RuntimeOverrideModel",
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
