"""DLKit - Deep Learning Toolkit.

A comprehensive toolkit for training, optimization, and inference
with machine learning models using Lightning, MLflow, and Optuna.
"""

import sys

# Surface primary application APIs
from .interfaces.api import (
    load_model,
    optimize,
    train,
    validate_config,
    search_registered_models,
    list_model_versions,
    get_model_version,
    register_logged_model,
    set_registered_model_alias,
    set_registered_model_version_tag,
    set_registered_model_version_tags,
    build_registered_model_uri,
    load_registered_model,
    LoggedModelRecord,
    search_logged_models,
    build_logged_model_uri,
    load_logged_model,
)

# Provide convenient access to configuration helpers and registries
from .tools.config.general_settings import GeneralSettings
from .tools.io import (
    load_config,
    load_raw_config,
    load_sections_config,
    load_section_config,
    reset_section_mappings,
    write_config,
    register_section_mapping,
    ConfigSectionError,
    ConfigValidationError,
)
from .tools.registry import (
    register_model,
    register_dataset,
    register_loss,
    register_metric,
    register_datamodule,
)

# Expose neural network modules for convenient access
from .core.models import nn

sys.modules[f"{__name__}.nn"] = nn


__all__ = [
    "train",
    "load_model",  # NEW: Stateful predictor API
    "optimize",
    "validate_config",
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
    "GeneralSettings",
    "load_config",
    "load_raw_config",
    "load_sections_config",
    "load_section_config",
    "reset_section_mappings",
    "write_config",
    "register_section_mapping",
    "ConfigSectionError",
    "ConfigValidationError",
    "register_model",
    "register_dataset",
    "register_loss",
    "register_metric",
    "register_datamodule",
    "nn",
]
