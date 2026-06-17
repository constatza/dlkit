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
from .geometry import FieldRole, FieldSpec, GeometryKind, GeometrySpec, TopologyKind
from .hooks import LifecycleHooks
from .overrides import ExecutionOverrides, OptimizationOverrides, TrainingOverrides
from .results import InferenceResult, OptimizationResult, TrainingResult
from .sources import ArraySource, EntryShapes, InputShapes, OutputShapes, Shape
from .state import ModelState

__all__ = [
    "ArraySource",
    "ConfigurationError",
    "DLKitError",
    "EntryShapes",
    "ExecutionOverrides",
    "FieldRole",
    "FieldSpec",
    "GeometryKind",
    "GeometrySpec",
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
    "Shape",
    "StrategyError",
    "TopologyKind",
    "TrainingResult",
    "TrainingOverrides",
    "WorkflowError",
]
