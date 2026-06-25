"""Configuration loading adapter for CLI."""

from __future__ import annotations

from pathlib import Path

from dlkit.common import ConfigurationError
from dlkit.infrastructure.config.factories import load_job
from dlkit.infrastructure.config.job_config import (
    InferenceJobConfig,
    SearchJobConfig,
    TrainingJobConfig,
)


def load_config(
    config_path: Path,
    run_type: str | None = None,
) -> TrainingJobConfig | InferenceJobConfig | SearchJobConfig:
    """Load configuration from file with CLI-specific error handling.

    Uses the new JobConfig hierarchy with ``load_job()`` as the primary loader.

    Args:
        config_path: Path to configuration file.
        run_type: Optional run type override (``"train"``, ``"predict"``, ``"search"``).
            Required when the TOML file does not include a ``[run] type`` key.

    Returns:
        Typed job config matched to the resolved ``run.type``.

    Raises:
        ConfigurationError: If configuration loading fails.
    """
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}", {"config_path": str(config_path)}
        )

    try:
        return load_job(config_path, run_type=run_type)
    except ValueError as e:
        raise ConfigurationError(
            f"Invalid configuration file: {e!s}",
            {"config_path": str(config_path), "error": str(e)},
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration: {e!s}",
            {"config_path": str(config_path), "error": str(e)},
        )


def validate_config_path(config_path: Path) -> bool:
    """Validate configuration file path and accessibility.

    Args:
        config_path: Path to configuration file

    Returns:
        True if valid; raises ConfigurationError if invalid.
    """
    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}", {"config_path": str(config_path)}
        )

    if not config_path.is_file():
        raise ConfigurationError(
            f"Configuration path is not a file: {config_path}", {"config_path": str(config_path)}
        )

    # Check file extension
    valid_extensions = {".toml", ".json", ".yaml", ".yml"}
    if config_path.suffix.lower() not in valid_extensions:
        raise ConfigurationError(
            f"Unsupported configuration file format: {config_path.suffix}. "
            f"Supported formats: {', '.join(valid_extensions)}",
            {"config_path": str(config_path), "extension": config_path.suffix},
        )

    try:
        # Test read permissions
        with open(config_path) as f:
            f.read(1)  # Read just one character to test access
    except PermissionError:
        raise ConfigurationError(
            f"Permission denied reading configuration file: {config_path}",
            {"config_path": str(config_path)},
        )
    except Exception as e:
        raise ConfigurationError(
            f"Cannot access configuration file: {e!s}",
            {"config_path": str(config_path), "error": str(e)},
        )
    return True
