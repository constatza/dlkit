"""DLKit - Deep Learning Toolkit.

A comprehensive toolkit for training, optimization, and inference
with machine learning models using Lightning, MLflow, and Optuna.
"""

from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import tomllib

# Surface primary application APIs
from .interfaces.api import load_predictor, optimize, train, validate_config

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
import sys
sys.modules[f"{__name__}.nn"] = nn


def _resolve_version() -> str:
    """Return the package version using pyproject as the source of truth."""
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project_file.is_file():
        try:
            project_data = tomllib.loads(project_file.read_text())
            return project_data.get("project", {}).get("version", "0.0.0")
        except Exception:
            pass
    try:
        return pkg_version("dlkit")
    except PackageNotFoundError:
        return "0.0.0"


__all__ = [
    "__version__",
    "train",
    "load_predictor",  # NEW: Stateful predictor API
    "optimize",
    "validate_config",
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


__version__ = _resolve_version()
