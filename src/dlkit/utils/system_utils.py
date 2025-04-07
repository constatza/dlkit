from types import ModuleType
from importlib import import_module
import os

from urllib3.util.url import parse_url
from pathlib import Path

from loguru import logger
from dlkit.utils.mlflow_utils import is_mlflow_server_running
import traceback

import traceback
from typing import NoReturn


def get_last_error_message(exc: Exception) -> str:
    tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return tb_lines[-1].strip()


def import_dynamic(module_path: str, prepend: str = "") -> type:
    """
    Dynamically import a module, class, function, or attribute from a string path.

    Args:
        module_path (str): The string path of the module, class, function, or attribute to import.
        prepend (str, optional): Optional string to prepend to the module path. Defaults to "".

    Returns:
        The imported module, class, function, or attribute.
    """
    # Replace potential path separators with dots
    module_path = module_path.replace("/", ".").replace("\\", ".")

    # Prepend optional path, if provided
    if prepend:
        module_path = f"{prepend}.{module_path}"

    try:
        module, attr = module_path.rsplit(".", 1)
        module: ModuleType = import_module(module)
        return getattr(module, attr)
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        last = get_last_error_message(e)
        logger.error(f"{last}")
        raise e


def ensure_local_directory(uri: str) -> None:
    """
    Ensures that the local directory for the given URI exists if the URI starts with `file:`.

    Args:
        uri (str): A URI that might be local (file scheme).
    """
    local_schemes = ["", "file"]
    parsed_uri = parse_url(uri)
    scheme = parsed_uri.scheme
    path = parsed_uri.path
    if path is None:
        return

    if os.name == "nt" and path.startswith("/"):
        path = path.lstrip("/")

    if scheme in local_schemes:
        local_path = Path(path)
        local_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured local directory exists: {local_path}")
