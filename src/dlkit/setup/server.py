from time import sleep
from subprocess import Popen
from pydantic import validate_call
from loguru import logger

from dlkit.settings import MLflowServerSettings
from dlkit.utils.system_utils import mkdir_if_local


@validate_call
def checks_before_start(config: MLflowServerSettings):
    """
    Start the MLflow server with validated configuration.

    Args:
        config (dict): Configuration dictionary parsed from a TOML file.
    """

    # Ensure directories exist if local
    mkdir_if_local(config.artifacts_destination)
    mkdir_if_local(config.backend_store_uri)
    logger.info(f"Backend store URI: {config.backend_store_uri}")
    logger.info(f"Artifacts destination: {config.artifacts_destination}")
    logger.info("Press Ctrl+C to stop.")


@validate_call
def start_server(config: MLflowServerSettings) -> Popen:
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
        "--artifacts-destination",
        config.artifacts_destination,
    ]

    checks_before_start(config)
    # Start in a new session on Unix
    mlflow_process = Popen(command, shell=False)
    return mlflow_process
