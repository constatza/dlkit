"""API functions module."""

from .core import train, optimize
from .config import validate_config, generate_template, validate_template
from .execution import execute
from .model_registry import (
    search_registered_models,
    list_model_versions,
    get_model_version,
    register_logged_model,
    set_registered_model_alias,
    set_registered_model_version_tag,
    set_registered_model_version_tags,
    build_registered_model_uri,
    load_registered_model,
)
from .model_logged import (
    LoggedModelRecord,
    search_logged_models,
    build_logged_model_uri,
    load_logged_model,
)

__all__ = [
    # Core workflow functions
    "train",
    "optimize",
    # Configuration functions
    "validate_config",
    "generate_template",
    "validate_template",
    # Unified execution function
    "execute",
    # Model registry helpers
    "search_registered_models",
    "list_model_versions",
    "get_model_version",
    "register_logged_model",
    "set_registered_model_alias",
    "set_registered_model_version_tag",
    "set_registered_model_version_tags",
    "build_registered_model_uri",
    "load_registered_model",
    "LoggedModelRecord",
    "search_logged_models",
    "build_logged_model_uri",
    "load_logged_model",
]
