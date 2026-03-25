"""Configuration persistence adapters for optimization results.

These adapters implement configuration storage following the Repository pattern,
allowing different storage formats and locations to be used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.runtime.workflows.optimization.domain import (
    IConfigurationPersistence,
    Study,
)
from dlkit.tools.utils.logging_config import get_logger

logger = get_logger(__name__)


class TOMLConfigurationPersister(IConfigurationPersistence):
    """TOML configuration persistence implementation.

    This persister saves optimization configurations in TOML format using
    the existing DLKit configuration writing utilities.
    """

    def __init__(self):
        """Initialize TOML persister."""

    def save_best_configuration(self, study: Study, configuration: dict[str, Any]) -> str | None:
        """Save best configuration to TOML file.

        Args:
            study: Study domain model
            configuration: Configuration to save

        Returns:
            Path to saved configuration file if successful, None otherwise
        """
        try:
            # Use environment-based output location
            from dlkit.tools.config import GeneralSettings
            from dlkit.tools.io import locations
            from dlkit.tools.io.config import write_config

            # Create output directory
            config_dir = locations.output("optuna_results")
            config_dir.mkdir(parents=True, exist_ok=True)

            # Generate config file name
            best_trial = study.best_trial
            trial_number = best_trial.trial_number if best_trial else 0
            config_path = (
                config_dir / f"best_config_study_{study.study_name}_trial_{trial_number}.toml"
            )

            # Convert configuration dict to GeneralSettings for TOML writing
            # This assumes the configuration is in the right format
            if isinstance(configuration, GeneralSettings):
                settings_to_save = configuration
            elif isinstance(configuration, dict):
                # Create settings from dict
                settings_to_save = GeneralSettings(**configuration)
            else:
                # Convert to dict first
                settings_to_save = GeneralSettings.model_validate(configuration)

            # Write configuration to TOML
            write_config(
                settings_to_save,
                config_path,
                exclude_value_entries=True,
            )

            logger.info("Saved best configuration to TOML at {}", config_path)

            return str(config_path)

        except Exception as e:
            logger.warning("Failed to save best configuration for '{}': {}", study.study_name, e)
            return None


class JSONConfigurationPersister(IConfigurationPersistence):
    """JSON configuration persistence implementation.

    This persister saves optimization configurations in JSON format.
    """

    def __init__(self):
        """Initialize JSON persister."""

    def save_best_configuration(self, study: Study, configuration: dict[str, Any]) -> str | None:
        """Save best configuration to JSON file.

        Args:
            study: Study domain model
            configuration: Configuration to save

        Returns:
            Path to saved configuration file if successful, None otherwise
        """
        try:
            import json

            from dlkit.tools.io import locations

            # Create output directory
            config_dir = locations.output("optuna_results")
            config_dir.mkdir(parents=True, exist_ok=True)

            # Generate config file name
            best_trial = study.best_trial
            trial_number = best_trial.trial_number if best_trial else 0
            config_path = (
                config_dir / f"best_config_study_{study.study_name}_trial_{trial_number}.json"
            )

            # Convert configuration to JSON-serializable format
            from pydantic import BaseModel as _BaseModel

            if isinstance(configuration, _BaseModel):
                config_dict = configuration.model_dump()
            elif isinstance(configuration, dict):
                config_dict = configuration
            else:
                config_dict = {"configuration": str(configuration)}

            # Write configuration to JSON
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

            logger.info("Saved best configuration to JSON at {}", config_path)

            return str(config_path)

        except Exception as e:
            logger.warning(
                "Failed to save best JSON configuration for '{}': {}",
                study.study_name,
                e,
            )
            return None


class NullConfigurationPersister(IConfigurationPersistence):
    """Null object implementation for when configuration persistence is disabled.

    This eliminates conditional logic by providing safe no-op behavior.
    """

    def save_best_configuration(self, study: Study, configuration: dict[str, Any]) -> str | None:
        """No-op configuration saving.

        Args:
            study: Study domain model (ignored)
            configuration: Configuration to save (ignored)

        Returns:
            None (no file saved)
        """
        return None


class FileSystemConfigurationPersister(IConfigurationPersistence):
    """File system configuration persistence with customizable format.

    This persister allows different file formats to be used based on
    file extension or explicit format specification.
    """

    def __init__(self, output_directory: Path | None = None, file_format: str = "toml"):
        """Initialize file system persister.

        Args:
            output_directory: Custom output directory
            file_format: File format ('toml', 'json', 'yaml')
        """
        self.output_directory = output_directory
        self.file_format = file_format.lower()

        # Create format-specific persisters
        self._persisters = {
            "toml": TOMLConfigurationPersister(),
            "json": JSONConfigurationPersister(),
        }

    def save_best_configuration(self, study: Study, configuration: dict[str, Any]) -> str | None:
        """Save best configuration using specified format.

        Args:
            study: Study domain model
            configuration: Configuration to save

        Returns:
            Path to saved configuration file if successful, None otherwise
        """
        persister = self._persisters.get(self.file_format)
        if not persister:
            logger.warning(
                "Unsupported configuration format '{}' (supported: {})",
                self.file_format,
                ", ".join(sorted(self._persisters.keys())),
            )
            return None

        # Use custom output directory if specified
        if self.output_directory:
            # Temporarily override the locations module behavior
            try:
                # This is a bit hacky but works for now
                # TODO: Improve this by making the persister take directory as parameter
                result = persister.save_best_configuration(study, configuration)
                if result and self.output_directory != Path(result).parent:
                    # Move file to custom directory
                    self.output_directory.mkdir(parents=True, exist_ok=True)
                    new_path = self.output_directory / Path(result).name
                    Path(result).rename(new_path)
                    return str(new_path)
                return result
            except Exception as e:
                logger.warning("Failed to save to custom directory: {}", e)
                return None
        else:
            return persister.save_best_configuration(study, configuration)
