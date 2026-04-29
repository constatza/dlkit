"""DLKit public package surface.

The package keeps top-level exports lazy so importing subpackages like
``dlkit.infrastructure.config`` does not trigger heavyweight model and inference imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ConfigSectionError": ("dlkit.infrastructure.io", "ConfigSectionError"),
    "ConfigValidationError": ("dlkit.infrastructure.io", "ConfigValidationError"),
    "GeneralSettings": ("dlkit.infrastructure.config.general_settings", "GeneralSettings"),
    "LoggedModelRecord": ("dlkit.interfaces.api", "LoggedModelRecord"),
    "execute": ("dlkit.interfaces.api", "execute"),
    "load_config": ("dlkit.infrastructure.io", "load_config"),
    "load_model": ("dlkit.interfaces.api", "load_model"),
    "load_raw_config": ("dlkit.infrastructure.io", "load_raw_config"),
    "load_section_config": ("dlkit.infrastructure.io", "load_section_config"),
    "load_sections_config": ("dlkit.infrastructure.io", "load_sections_config"),
    "register_datamodule": ("dlkit.infrastructure.registry", "register_datamodule"),
    "register_dataset": ("dlkit.infrastructure.registry", "register_dataset"),
    "register_loss": ("dlkit.infrastructure.registry", "register_loss"),
    "register_metric": ("dlkit.infrastructure.registry", "register_metric"),
    "register_model": ("dlkit.infrastructure.registry", "register_model"),
    "register_section_mapping": ("dlkit.infrastructure.io", "register_section_mapping"),
    "reset_section_mappings": ("dlkit.infrastructure.io", "reset_section_mappings"),
    "validate_checkpoint": ("dlkit.interfaces.inference", "validate_checkpoint"),
    "get_checkpoint_info": ("dlkit.interfaces.inference", "get_checkpoint_info"),
    "validate_config": ("dlkit.interfaces.api", "validate_config"),
    "write_config": ("dlkit.infrastructure.io", "write_config"),
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
