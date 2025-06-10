"""
This module provides functions for loading and validating settings from a configuration file.
"""

from dynaconf import Dynaconf
from dynaconf.utils.parse_conf import DynaconfFormatError
from loguru import logger
from pydantic import validate_call, FilePath

from dlkit.settings.general_settings import Settings
from dlkit.settings.utils import dynaconf_to_settings


@validate_call
def load_validated_settings(path: FilePath) -> Settings:
    """Load and validate settings from a configuration file.

    This function reads a configuration file specified by `file_path` using
    Dynaconf, validates it, and converts it into a `Settings` object. If the
    configuration file is not valid, a `DynaconfFormatError` is raised.

    Args:
        path (FilePath): The path to the configuration file.

    Returns:
        Settings: A validated `Settings` object.

    Raises:
        DynaconfFormatError: If the configuration file format is invalid.
    """
    try:
        config: Dynaconf = Dynaconf(
            root_path=path.parent,
            settings_files=[path.name],
            envvar_prefix="DLKIT",
            load_dotenv=True,
        )
    except DynaconfFormatError as e:
        logger.error(f"Configuration file is not valid - {str(path)}")
        raise ValueError(f"Configuration file is not valid - {str(path)}") from e
    config.paths["settings"] = str(path)
    dlkit_settings = dynaconf_to_settings(config)
    return dlkit_settings
