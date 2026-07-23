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
    ChildFailure,
    ChildOutcome,
    ChildSuccess,
    ConvergencePoint,
    ConvergenceResult,
    EvaluationResult,
    FailurePolicy,
    InferenceResult,
    MultiRunResult,
    OptimizationResult,
    TrainingResult,
    WorkflowResult,
)
from .shapes import InputShapes, OutputShapes, Shape, ShapeContext, ShapeProvider
from .sources import ArraySource
from .state import ModelState

__all__ = [
    "ArraySource",
    "BatchComplianceError",
    "ChildFailure",
    "ChildOutcome",
    "ChildSuccess",
    "ConfigurationError",
    "ConvergencePoint",
    "ConvergenceResult",
    "DLKitError",
    "EvaluationResult",
    "ExecutionOverrides",
    "FailurePolicy",
    "InferenceResult",
    "InputShapes",
    "LifecycleHooks",
    "ModelLoadingError",
    "ModelState",
    "ModelStateError",
    "MultiRunResult",
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
    "WorkflowResult",
]
