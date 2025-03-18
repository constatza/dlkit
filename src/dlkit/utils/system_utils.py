import functools
import importlib
import os
import signal
import subprocess
import sys
import socket

from urllib3.util.url import parse_url
from pathlib import Path

from loguru import logger
from dlkit.utils.mlflow_utils import is_mlflow_server_running


def import_dynamically(module_path: str, prepend: str = ""):
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

    # Split the path into components
    path_parts = module_path.split(".")
    module = importlib.import_module(".".join(path_parts[:-1]))

    # If the last part is not found, try importing as an attribute (class, function, etc.)
    try:
        attr = getattr(module, path_parts[-1])
        return attr
    except AttributeError as e:
        # If it's not an attribute, return the full module instead
        raise e


def filter_kwargs(kwargs: dict):
    """
    Filter keyword arguments to only include valid parameters for a class constructor
    and return a new instance of the class with the filtered keyword arguments.
    """
    # sig = inspect.signature(cls.__init__)
    # # Get valid argument names (excluding 'self')
    # valid_params = set(sig.parameters) - {"self"}

    # Filter kwargs to only include valid parameters
    return {k: v for k, v in kwargs.items() if k != "name"}


def check_port_available(host, port, terminate_apps_on_port=False):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        try:
            sock.bind((host, port))
        except socket.error:
            logger.warn(f"Port {port} is already in use.")

            if is_mlflow_server_running(host, port):
                logger.warn("MLflow server is already running on the port.")
                return

            if terminate_apps_on_port:
                logger.warn("Terminating applications using the port.")
                terminate_apps(port)
                check_port_available(host, port, terminate_apps_on_port=False)
            else:
                logger.warn(
                    "Please specify a different port or terminate the applications using the port."
                )
                sys.exit(1)


def terminate_apps(port):
    if os.name == "nt":
        # kill windows apps using port
        subprocess.run(
            f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr "{port}"\') do taskkill /f /pid %a',
            shell=True,
            stdout=subprocess.DEVNULL,
        )

    else:
        subprocess.run(
            f"kill -9 $(lsof -t -i:{port})", shell=True, stdout=subprocess.DEVNULL
        )


def cleanup(server_process):
    """Terminate the MLflow server process group."""
    if server_process.poll() is None:  # Server still running
        if os.name != "nt":
            # On Unix, kill the entire process group
            os.killpg(server_process.pid, signal.SIGTERM)
        else:
            # On Windows, just terminate the process (created in new group)
            server_process.terminate()


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
