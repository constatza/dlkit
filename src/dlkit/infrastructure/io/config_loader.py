"""TOML config loading functions for lowercase JobConfig workflows."""

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from dlkit.infrastructure.io.config_errors import ConfigValidationError

if TYPE_CHECKING:
    from dlkit.infrastructure.config.job_config import (
        InferenceJobConfig,
        SearchJobConfig,
        TrainingJobConfig,
    )


def load_config[T: BaseModel](
    config_path: Path | str,
    model_class: type[T] | None = None,
    raw: bool = False,
) -> T | dict[str, Any]:
    """Load a TOML config file.

    By default, loads the config as raw dict. Use model_class to specify validation.

    Args:
        config_path: Path to the TOML config file
        model_class: Pydantic model class to validate the config (None for raw dict)
        raw: If True, return raw config dict without validation

    Returns:
        Raw dict by default, specified model_class if provided
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    from dlkit.infrastructure.config.core.sources import DLKitTomlSource

    source = DLKitTomlSource(config_path)
    config_data = source()

    # Return raw dict if explicitly requested
    if raw:
        return config_data

    # Return raw dict if no model class specified
    if model_class is None:
        return config_data
    resolved_model_class: type[BaseModel] = model_class

    try:
        validated = resolved_model_class.model_validate(config_data)
    except Exception as e:
        raise ConfigValidationError(
            f"Failed to validate config with {resolved_model_class.__name__}: {e}",
            resolved_model_class.__name__,
            config_data,
        ) from e

    return validated


def load_raw_config(config_path: Path | str) -> dict[str, Any]:
    """Load raw TOML config file as dict without validation.

    Args:
        config_path: Path to the TOML config file

    Returns:
        Raw config dictionary
    """
    return load_config(config_path, raw=True)


def check_section_exists(config_path: Path | str, section_name: str) -> bool:
    """Check if a section exists in the config file without full parsing.

    Args:
        config_path: Path to the TOML config file
        section_name: Name of section to check

    Returns:
        True if section exists, False otherwise

    Raises:
        FileNotFoundError: If config file doesn't exist

    """
    return section_name in get_available_sections(config_path)


def get_available_sections(config_path: Path | str) -> list[str]:
    """Get list of available sections without full config parsing.

    Args:
        config_path: Path to the TOML config file

    Returns:
        List of section names found in the config

    Raises:
        FileNotFoundError: If config file doesn't exist

    Example:
        >>> sections = get_available_sections("config.toml")
        >>> isinstance(sections, list)
        True
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return list(tomllib.load(f).keys())


def load_training_config_eager(config_path: Path | str) -> TrainingJobConfig:
    """Load training config with eager Pydantic validation.

    Validates the TOML immediately (fail-fast on typos/types). Uses the new
    ``TrainingJobConfig`` with lowercase section keys.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        TrainingJobConfig with eagerly validated sections.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config has type errors or invalid values.
    """
    from dlkit.infrastructure.config.job_config import TrainingJobConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    toml_data = load_config(config_path, raw=True)
    return TrainingJobConfig.model_validate(toml_data)


def load_inference_config_eager(config_path: Path | str) -> InferenceJobConfig:
    """Load inference config with eager Pydantic validation.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        InferenceJobConfig with eagerly validated sections.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config has type errors or invalid values.
    """
    from dlkit.infrastructure.config.job_config import InferenceJobConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    toml_data = load_config(config_path, raw=True)
    return InferenceJobConfig.model_validate(toml_data)


def load_optimization_config_eager(config_path: Path | str) -> SearchJobConfig:
    """Load search/optimization config with eager Pydantic validation.

    Args:
        config_path: Path to TOML configuration file.

    Returns:
        SearchJobConfig with eagerly validated sections.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config has type errors or invalid values.
    """
    from dlkit.infrastructure.config.job_config import SearchJobConfig

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    toml_data = load_config(config_path, raw=True)
    return SearchJobConfig.model_validate(toml_data)
