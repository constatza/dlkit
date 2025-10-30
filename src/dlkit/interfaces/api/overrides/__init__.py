"""Runtime parameter override system for DLKit API.

This module provides functionality to apply runtime parameter overrides
to GeneralSettings objects while maintaining Pydantic model integrity
and type safety.
"""

from .manager import BasicOverrideManager, basic_override_manager
from .normalizer import OverrideNormalizer
from .types import BasicOverrides, MLflowOverrides, PathOverrides, TrainingOverrides

__all__ = [
    # Override manager
    "BasicOverrideManager",
    "basic_override_manager",
    # Override normalizer
    "OverrideNormalizer",
    # Type definitions
    "BasicOverrides",
    "MLflowOverrides",
    "PathOverrides",
    "TrainingOverrides",
]
