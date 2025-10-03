"""Runtime parameter override system for DLKit API.

This module provides functionality to apply runtime parameter overrides
to GeneralSettings objects while maintaining Pydantic model integrity
and type safety.
"""

from .manager import BasicOverrideManager, basic_override_manager
from .types import BasicOverrides, MLflowOverrides, PathOverrides, TrainingOverrides

__all__ = [
    # Override manager
    "BasicOverrideManager",
    "basic_override_manager",
    # Type definitions
    "BasicOverrides",
    "MLflowOverrides",
    "PathOverrides",
    "TrainingOverrides",
]
