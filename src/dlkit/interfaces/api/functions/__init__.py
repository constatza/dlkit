"""API functions module."""

from .config import generate_template, validate_config, validate_template
from .core import build_inference_datamodule, optimize, train
from .execution import execute
from .model_logged import (
    LoggedModelRecord,
    build_logged_model_uri,
    load_logged_model,
    search_logged_models,
)
from .model_registry import (
    build_registered_model_uri,
    get_model_version,
    list_model_versions,
    load_registered_model,
    register_logged_model,
    search_registered_models,
    set_registered_model_alias,
    set_registered_model_version_tag,
    set_registered_model_version_tags,
)

__all__ = [
    # Core workflow functions
    "train",
    "optimize",
    "build_inference_datamodule",
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
