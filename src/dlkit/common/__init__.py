"""Shared cross-layer contracts for DLKit."""

from .errors import (
    BatchComplianceError,
    ConfigurationError,
    DLKitError,
    ModelLoadingError,
    ModelStateError,
    PluginError,
    StrategyError,
    WorkflowError,
)
from .hooks import LifecycleHooks, RunCreatedEvent, RunKind
from .overrides import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from .results import (
    ConvergencePoint,
    ConvergenceResult,
    InferenceResult,
    OptimizationResult,
    TrainingResult,
)
from .shapes import InputShapes, OutputShapes, Shape, ShapeContext, ShapeProvider
from .sources import ArraySource
from .state import ModelState

__all__ = [
    "ArraySource",
    "BatchComplianceError",
    "ConfigurationError",
    "ConvergencePoint",
    "ConvergenceResult",
    "DLKitError",
    "ExecutionOverrides",
    "InferenceResult",
    "InputShapes",
    "LifecycleHooks",
    "ModelLoadingError",
    "ModelState",
    "ModelStateError",
    "OptimizationResult",
    "OptimizationOverrides",
    "OutputShapes",
    "PluginError",
    "RunCreatedEvent",
    "RunKind",
    "Shape",
    "ShapeContext",
    "ShapeProvider",
    "StrategyError",
    "TrainingResult",
    "TrainingOverrides",
    "WorkflowError",
]
