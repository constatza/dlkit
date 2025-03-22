from dynaconf import Dynaconf, LazySettings
from pydantic import validate_call, FilePath
from dlkit.settings.general_settings import Settings
from dlkit.settings.utils import dynaconf_to_settings


@validate_call
def load_config(config_path: FilePath) -> LazySettings:

    # Load the TOML config that uses interpolation (without forcing everything under a default namespace)
    """
    Loads a configuration from a TOML file, using Dynaconf to handle interpolation and environment variables.

    Args:
        config_path (FilePath): The path to the TOML configuration file.

    Returns:
        dict: A dictionary representation of the configuration with keys in lowercase.
    """
    config = Dynaconf(settings_files=[config_path], envvar_prefix="DLKIT")
    return config


# @validate_call
def load_settings_from(file_path: FilePath) -> Settings:
    config = Dynaconf(settings_files=[file_path], envvar_prefix="DLKIT")
    dlkit_settings = dynaconf_to_settings(config)
    return dlkit_settings
