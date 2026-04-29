"""Shared cross-layer contracts for DLKit."""

from .errors import (
    ConfigurationError,
    DLKitError,
    ModelLoadingError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)
from .hooks import LifecycleHooks
from .overrides import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from .results import InferenceResult, OptimizationResult, TrainingResult
from .shapes import ShapeSpecProtocol, ShapeSummary
from .state import ModelState

__all__ = [
    "ConfigurationError",
    "DLKitError",
    "ExecutionOverrides",
    "InferenceResult",
    "LifecycleHooks",
    "ModelLoadingError",
    "ModelState",
    "ModelStateError",
    "OptimizationResult",
    "OptimizationOverrides",
    "PluginError",
    "ShapeSpecProtocol",
    "ShapeSummary",
    "StrategyError",
    "TrainingResult",
    "TrainingOverrides",
    "WorkflowError",
]
