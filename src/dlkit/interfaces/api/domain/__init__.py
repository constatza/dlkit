"""DLKit API domain layer.

This module contains the core domain models, protocols, and errors
that define the business logic of DLKit without framework dependencies.
"""

from .errors import (
    ConfigurationError,
    DLKitError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)
from .models import InferenceResult, ModelState, OptimizationResult, TrainingResult
from .precision import (
    PrecisionContext,
    PrecisionProvider,
    precision_override,
    get_precision_context,
)
from .protocols import ExecutionStrategy, StrategyFactory, WorkflowOperation

__all__ = [
    # Models
    "ModelState",
    "TrainingResult",
    "InferenceResult",
    "OptimizationResult",
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
