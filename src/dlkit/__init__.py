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
    "build_logged_model_uri": ("dlkit.interfaces.api", "build_logged_model_uri"),
    "build_registered_model_uri": ("dlkit.interfaces.api", "build_registered_model_uri"),
    "get_model_version": ("dlkit.interfaces.api", "get_model_version"),
    "list_model_versions": ("dlkit.interfaces.api", "list_model_versions"),
    "load_config": ("dlkit.infrastructure.io", "load_config"),
    "load_logged_model": ("dlkit.interfaces.api", "load_logged_model"),
    "load_model": ("dlkit.interfaces.api", "load_model"),
    "load_raw_config": ("dlkit.infrastructure.io", "load_raw_config"),
    "load_registered_model": ("dlkit.interfaces.api", "load_registered_model"),
    "load_section_config": ("dlkit.infrastructure.io", "load_section_config"),
    "load_sections_config": ("dlkit.infrastructure.io", "load_sections_config"),
    "optimize": ("dlkit.interfaces.api", "optimize"),
    "register_datamodule": ("dlkit.infrastructure.registry", "register_datamodule"),
    "register_dataset": ("dlkit.infrastructure.registry", "register_dataset"),
    "register_logged_model": ("dlkit.interfaces.api", "register_logged_model"),
    "register_loss": ("dlkit.infrastructure.registry", "register_loss"),
    "register_metric": ("dlkit.infrastructure.registry", "register_metric"),
    "register_model": ("dlkit.infrastructure.registry", "register_model"),
    "register_section_mapping": ("dlkit.infrastructure.io", "register_section_mapping"),
    "reset_section_mappings": ("dlkit.infrastructure.io", "reset_section_mappings"),
    "search_logged_models": ("dlkit.interfaces.api", "search_logged_models"),
    "search_registered_models": ("dlkit.interfaces.api", "search_registered_models"),
    "set_registered_model_alias": ("dlkit.interfaces.api", "set_registered_model_alias"),
    "set_registered_model_version_tag": (
        "dlkit.interfaces.api",
        "set_registered_model_version_tag",
    ),
    "set_registered_model_version_tags": (
        "dlkit.interfaces.api",
        "set_registered_model_version_tags",
    ),
    "train": ("dlkit.interfaces.api", "train"),
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
