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
from .state import ModelState

__all__ = [
    "ConfigurationError",
    "DLKitError",
    "ExecutionOverrides",
    "FieldRole",
    "FieldSpec",
    "GeometryKind",
    "GeometrySpec",
    "InferenceResult",
    "LifecycleHooks",
    "ModelLoadingError",
    "ModelState",
    "ModelStateError",
    "OptimizationResult",
    "OptimizationOverrides",
    "PluginError",
    "StrategyError",
    "TopologyKind",
    "TrainingResult",
    "TrainingOverrides",
    "WorkflowError",
]
