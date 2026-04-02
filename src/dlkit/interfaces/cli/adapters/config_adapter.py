"""Configuration loading adapter for CLI."""

from __future__ import annotations

from pathlib import Path

from dlkit.shared import ConfigurationError
from dlkit.tools.config import load_settings
from dlkit.tools.config.protocols import BaseSettingsProtocol
from dlkit.tools.io.path_context import path_override_context


def load_config(
    config_path: Path,
    *,
    root_dir: Path | None = None,
    output_dir: Path | None = None,
    workflow_type: str | None = None,
) -> BaseSettingsProtocol:
    """Load configuration from file with CLI-specific error handling and partial loading.

    Uses the new SOLID-compliant configuration system with protocol-based dependency inversion.
    Supports workflow-specific partial loading for optimal performance.

    Args:
        config_path: Path to configuration file
        root_dir: Optional root directory override
        output_dir: Optional output directory override (deprecated, use path context)
        workflow_type: Workflow type for partial loading ('training', 'inference', or None for all sections)

    Returns:
        BaseSettingsProtocol: Loaded settings object appropriate for the workflow type

    Raises:
        ConfigurationError: If configuration loading fails
    """
    try:
        # Validate config file exists
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}", {"config_path": str(config_path)}
            )

        # Load settings - all workflows use the same loading function
        load_fn = load_settings

        # Load settings from file. If CLI provides root_dir, apply as a temporary context.
        if root_dir is not None:
            with path_override_context({"root_dir": root_dir}):
                settings = load_fn(config_path)
        else:
            settings = load_fn(config_path)

        # Note: Path overrides are applied via a path context and not mutated into settings.

        return settings

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
