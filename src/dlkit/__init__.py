"""DLKit public package surface.

The package keeps top-level exports lazy so importing subpackages like
``dlkit.infrastructure.config`` does not trigger heavyweight model and inference imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "execute": ("dlkit.interfaces.api", "execute"),
    "load_inference_config": ("dlkit.config", "load_inference_config"),
    "load_model": ("dlkit.inference", "load_model"),
    "load_optimization_config": ("dlkit.config", "load_optimization_config"),
    "load_training_config": ("dlkit.config", "load_training_config"),
    "optimize": ("dlkit.interfaces.api", "optimize"),
    "register_dataset": ("dlkit.registry", "register_dataset"),
    "register_model": ("dlkit.registry", "register_model"),
    "train": ("dlkit.interfaces.api", "train"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve top-level exports lazily."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy exports in interactive tooling."""
    return sorted(__all__)
