"""Core settings infrastructure for DLKit.

This module provides the foundation for the settings system using SOLID principles:
- Factory pattern for object construction
- Registry pattern for dynamic class resolution
- Context pattern for dependency injection
"""

from .base_settings import BasicSettings, ComponentSettings, HyperParameterSettings
from .context import BuildContext
from .factories import ComponentFactory, ComponentRegistry, FactoryProvider
from .patching import apply_patch, compile_mixed_overrides, patch_model

__all__ = [
    "BasicSettings",
    "BuildContext",
    "ComponentFactory",
    "ComponentRegistry",
    "ComponentSettings",
    "FactoryProvider",
    "HyperParameterSettings",
    "apply_patch",
    "compile_mixed_overrides",
    "patch_model",
]
