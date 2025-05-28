"""
This module provides functions for loading and validating settings from a configuration file.
"""

from dynaconf import Dynaconf
from dynaconf.utils.parse_conf import DynaconfFormatError
from loguru import logger
from pydantic import FilePath, validate_call

from dlkit.settings.general_settings import Settings
from dlkit.settings.utils import dynaconf_to_settings


@validate_call
def load_validated_settings(file_path: FilePath) -> Settings:
    """Load and validate settings from a configuration file.

    This function reads a configuration file specified by `file_path` using
    Dynaconf, validates it, and converts it into a `Settings` object. If the
    configuration file is not valid, a `DynaconfFormatError` is raised.

    Args:
        file_path (FilePath): The path to the configuration file.

    Returns:
        Settings: A validated `Settings` object.

    Raises:
        DynaconfFormatError: If the configuration file format is invalid.
    """
    try:
        config: Dynaconf = Dynaconf(settings_files=[file_path], envvar_prefix="DLKIT")
    except DynaconfFormatError as e:
        logger.error(f"Configuration file is not valid - {str(file_path)}")
        raise ValueError(f"Configuration file is not valid - {str(file_path)}") from e
    config.paths["settings"] = file_path
    dlkit_settings = dynaconf_to_settings(config)
    return dlkit_settings
