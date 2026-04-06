"""DLKit API package public surface."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ConfigurationError": ("dlkit.common", "ConfigurationError"),
    "InferenceResult": ("dlkit.common", "InferenceResult"),
    "LoggedModelRecord": ("dlkit.interfaces.api.functions", "LoggedModelRecord"),
    "OptimizationResult": ("dlkit.common", "OptimizationResult"),
    "StrategyError": ("dlkit.common", "StrategyError"),
    "LifecycleHooks": ("dlkit.common", "LifecycleHooks"),
    "TrainingResult": ("dlkit.common", "TrainingResult"),
    "WorkflowError": ("dlkit.common", "WorkflowError"),
    "build_logged_model_uri": ("dlkit.interfaces.api.functions", "build_logged_model_uri"),
    "build_registered_model_uri": (
        "dlkit.interfaces.api.functions",
        "build_registered_model_uri",
    ),
    "execute": ("dlkit.interfaces.api.functions", "execute"),
    "generate_template": ("dlkit.interfaces.api.functions", "generate_template"),
    "get_checkpoint_info": ("dlkit.interfaces.inference", "get_checkpoint_info"),
    "get_model_version": ("dlkit.interfaces.api.functions", "get_model_version"),
    "list_model_versions": ("dlkit.interfaces.api.functions", "list_model_versions"),
    "load_logged_model": ("dlkit.interfaces.api.functions", "load_logged_model"),
    "load_model": ("dlkit.interfaces.inference", "load_model"),
    "load_registered_model": ("dlkit.interfaces.api.functions", "load_registered_model"),
    "optimize": ("dlkit.interfaces.api.functions", "optimize"),
    "register_logged_model": ("dlkit.interfaces.api.functions", "register_logged_model"),
    "search_logged_models": ("dlkit.interfaces.api.functions", "search_logged_models"),
    "search_registered_models": ("dlkit.interfaces.api.functions", "search_registered_models"),
    "set_registered_model_alias": (
        "dlkit.interfaces.api.functions",
        "set_registered_model_alias",
    ),
    "set_registered_model_version_tag": (
        "dlkit.interfaces.api.functions",
        "set_registered_model_version_tag",
    ),
    "set_registered_model_version_tags": (
        "dlkit.interfaces.api.functions",
        "set_registered_model_version_tags",
    ),
    "train": ("dlkit.interfaces.api.functions", "train"),
    "validate_checkpoint": ("dlkit.interfaces.inference", "validate_checkpoint"),
    "validate_config": ("dlkit.interfaces.api.functions", "validate_config"),
    "validate_template": ("dlkit.interfaces.api.functions", "validate_template"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve API exports lazily to avoid import cycles."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy exports in interactive tooling."""
    return sorted(__all__)
