"""
This module provides functions for loading and validating settings from a configuration file.
"""

from dynaconf import Dynaconf
from dynaconf.utils.parse_conf import DynaconfFormatError
from loguru import logger
from pydantic import validate_call, DirectoryPath, FilePath

from dlkit.settings.general_settings import Settings
from dlkit.settings.utils import dynaconf_to_settings


@validate_call
def load_validated_settings(settings_dir: DirectoryPath | FilePath) -> Settings:
    """Load and validate settings from a configuration file.

    This function reads a configuration file specified by `file_path` using
    Dynaconf, validates it, and converts it into a `Settings` object. If the
    configuration file is not valid, a `DynaconfFormatError` is raised.

    Args:
        settings_dir (FilePath): The path to the configuration file.

    Returns:
        Settings: A validated `Settings` object.

    Raises:
        DynaconfFormatError: If the configuration file format is invalid.
    """
    try:
        if settings_dir.is_file():
            settings_dir = settings_dir.parent
        config: Dynaconf = Dynaconf(
            root_path=settings_dir,
            settings_files=["*.toml"],
            envvar_prefix="DLKIT",
            load_dotenv=True,
        )
    except DynaconfFormatError as e:
        logger.error(f"Configuration file is not valid - {str(settings_dir)}")
        raise ValueError(f"Configuration file is not valid - {str(settings_dir)}") from e
    config.paths["settings_dir"] = settings_dir
    dlkit_settings = dynaconf_to_settings(config)
    return dlkit_settings
