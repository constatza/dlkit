from types import ModuleType
from importlib import import_module
import os

from urllib3.util.url import parse_url
from pathlib import Path

from loguru import logger

import traceback


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
        module_name, class_name = module_path.rsplit(".", 1)
        module: ModuleType = import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ModuleNotFoundError) as e:
        last = get_last_error_message(e)
        logger.error(f"{last}")
        raise e


def is_local_host(hostname: str | None) -> bool:
    """
    Check if a hostname is considered local.

    A URI is considered local if:
      - It has no hostname (which usually means a file path or similar local resource),
      - Its hostname is 'localhost' (case-insensitive),
      - Its hostname is the loopback IPv4 address ('127.0.0.1'),
      - Its hostname is a common representation of the IPv6 loopback address ('::1').

    Args:
        hostname (str | None): The hostname to check. Defaults to None.

    Returns:
        bool: True if the URI is considered local, otherwise False.
    """
    if hostname is None:
        return True
    hostname = hostname.lower()
    loopbacks = {"", "0.0.0.0", "127.0.0.0.1", "::1"}
    # Check known local identifiers
    if hostname in loopbacks:
        return True
    return False


def mkdir_if_local(uri: str) -> None:
    """
    Ensures that the local directory for the given URI exists if the URI starts with `file:`.

    Args:
        uri (str): A URI that might be local (file scheme).
    """
    parsed_uri = parse_url(uri)
    path = parsed_uri.path
    if not is_local_host(parsed_uri.hostname):
        logger.warning(
            f"URI: {uri} does not appear to be local. Skipping directory creation."
        )
        return

    if os.name == "nt" and path.startswith("/"):
        path = path.lstrip("/")

    local_path = Path(path)
    # check if it has extension
    if not local_path.suffix:
        local_path = local_path.parent
    local_path.mkdir(parents=True, exist_ok=True)
