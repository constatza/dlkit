import argparse
from dynaconf import Dynaconf
from typing import Any
from pydantic import validate_call, FilePath


@validate_call
def load_config(config_path: FilePath) -> dict:

    # Load the TOML config that uses interpolation (without forcing everything under a default namespace)
    """
    Loads a configuration from a TOML file, using Dynaconf to handle interpolation and environment variables.

    Args:
        config_path (FilePath): The path to the TOML configuration file.

    Returns:
        dict: A dictionary representation of the configuration with keys in lowercase.
    """
    config = Dynaconf(settings_files=[config_path], load_dotenv=True)
    return config.to_dict(lower=True)


def parse_config(description: str = "") -> dict[str, Any]:
    """
    Parse the command line arguments and load a configuration from a TOML file.

    Args:
        description (str): The description of the command line interface.

    Returns:
        dict[str, Any]: A dictionary representation of the configuration with keys in lowercase.
    """
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument(
        "config", type=str, help="Path to the TOML configuration file."
    )
    config_path = argparser.parse_args().config
    return load_config(config_path)
