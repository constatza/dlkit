"""Shared cross-layer contracts for DLKit."""

from .errors import (
    BatchComplianceError,
    ConfigurationError,
    DLKitError,
    ModelLoadingError,
    ModelStateError,
    PluginError,
    StrategyError,
    TrackingError,
    WorkflowError,
)
from .hooks import (
    ChildPlannedEvent,
    LifecycleHooks,
    RunCreatedEvent,
    RunKind,
    SweepCompletedEvent,
)
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
    "ChildPlannedEvent",
    "ChildSuccess",
    "ConfigurationError",
    "ConvergencePoint",
    "ConvergenceResult",
    "DLKitError",
    "EvaluationResult",
    "FailurePolicy",
    "InferenceResult",
    "InputShapes",
    "LifecycleHooks",
    "ModelLoadingError",
    "ModelState",
    "ModelStateError",
    "MultiRunResult",
    "OptimizationResult",
    "OutputShapes",
    "PluginError",
    "RunCreatedEvent",
    "RunKind",
    "Shape",
    "ShapeContext",
    "ShapeProvider",
    "StrategyError",
    "SweepCompletedEvent",
    "TrackingError",
    "TrainingResult",
    "WorkflowError",
    "WorkflowResult",
]
