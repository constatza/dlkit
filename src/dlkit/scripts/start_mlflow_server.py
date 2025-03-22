import os
import signal
import subprocess
import sys
import atexit
import click
from pydantic import validate_call, FilePath
from loguru import logger
from dlkit.utils.system_utils import (
    ensure_local_directory,
    check_port_available,
)
from dlkit.settings import MLflowServerSettings
from dlkit.io.readers import load_settings_from


def kill_process_group(popen: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    """
    Sends a termination signal to the entire process group associated with the given subprocess.

    Args:
        popen (subprocess.Popen): The subprocess whose process group should be terminated.
        sig (int): The signal to send (default is SIGTERM).
    """
    if os.name == "nt":
        # On Windows, terminate the process (created in new group)
        popen.terminate()
        return
    try:
        # Get the process group ID of the subprocess and send the signal to all processes in that group.
        os.killpg(os.getpgid(popen.pid), sig)
    except ProcessLookupError:
        # Process or process group already terminated.
        pass


def register_exit_handlers(popen: subprocess.Popen) -> None:
    """
    Registers handlers to ensure that the MLflow server process group is terminated when the parent process exits.

    This function registers both atexit and signal handlers for SIGTERM and SIGINT.

    Args:
        popen (subprocess.Popen): The subprocess representing the MLflow server.
    """

    def cleanup(*args) -> None:
        kill_process_group(popen)
        # Exit immediately after cleanup.
        # sys.exit(0)

    # Register cleanup on normal interpreter shutdown.
    atexit.register(cleanup)
    # Register cleanup on SIGTERM and SIGINT (e.g., when the user presses Ctrl+C)
    signal.signal(signal.SIGTERM, lambda signum, frame: cleanup())
    signal.signal(signal.SIGINT, lambda signum, frame: cleanup())


def checks_before_start(config: MLflowServerSettings):
    """
    Start the MLflow server with validated configuration.

    Args:
        config (dict): Configuration dictionary parsed from a TOML file.
    """

    # Initialize MLflow server settings (e.g., creating or retrieving the experiment)
    check_port_available(config.host, config.port, config.terminate_apps_on_port)
    # Ensure directories exist if local
    ensure_local_directory(config.default_artifact_root)
    ensure_local_directory(config.backend_store_uri)

    logger.info(f"Backend store URI: {config.backend_store_uri}")
    logger.info(f"Default artifact root: {config.default_artifact_root}")
    logger.info("Press Ctrl+C to stop.")


def popen_mlflow_server(config: MLflowServerSettings) -> subprocess.Popen:
    """
    Starts the MLflow server as a subprocess.

    Args:
        config:
            host (str): Host address for the MLflow server.
            port (int): Port for the MLflow server.
            backend_store_uri (str): Backend store URI for MLflow tracking.
            artifact_root (str): Default artifact root directory or URI.
    """
    command = [
        "mlflow",
        "server",
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--backend-store-uri",
        config.backend_store_uri,
        "--default-artifact-root",
        config.default_artifact_root,
    ]
    # Start in a new session on Unix
    mlflow_process = subprocess.Popen(command, shell=False)
    logger.info(f"MLflow server started with PID: {mlflow_process.pid}")
    # Register cleanup handlers.
    register_exit_handlers(mlflow_process)

    return mlflow_process


@click.command("Start MLflow Server")
@click.argument("config_path", default="./config.toml")
@logger.catch
@validate_call
def main(config_path: FilePath):
    try:
        settings = load_settings_from(config_path)
        checks_before_start(settings.MLFLOW.server)
        mlflow_server = popen_mlflow_server(settings.MLFLOW.server)
        mlflow_server.wait()
    except Exception as e:
        logger.error(e)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
