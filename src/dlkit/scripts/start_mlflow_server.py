from dlkit.io.logging import get_logger
from dlkit.io.readers import read_toml
from dlkit.setup.tracking import MLFlowServerConfig
from dlkit.utils.system_utils import (
    ensure_local_directory,
    cleanup,
    check_port_available,
)

import subprocess
import argparse
import os
import sys
import signal
import atexit

logger = get_logger(__name__)


def popen_server(config: MLFlowServerConfig):
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
    if os.name == "nt":
        # Create a new process group on Windows
        return subprocess.Popen(
            command, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, shell=False
        )

        # Start in a new session on Unix
    return subprocess.Popen(command, preexec_fn=os.setpgrp, shell=False)


def start_mlflow_server(config: MLFlowServerConfig):
    """
    Start the MLflow server with validated configuration.

    Args:
        config (dict): Configuration dictionary parsed from a TOML file.
    """

    # Initialize MLflow server settings (e.g., creating or retrieving the experiment)
    check_port_available(config.host, config.port, config.terminate_apps_on_port)
    # Ensure directories exist if local
    ensure_local_directory(config.tracking_uri)
    ensure_local_directory(config.default_artifact_root)

    # Set the MLflow tracking URI for the server
    server_process = popen_server(config)
    atexit.register(cleanup, server_process)
    logger.info(f"MLflow server started with PID: {server_process.pid}")

    logger.info(f"Port: {config.port}")
    logger.info(f"Backend store URI: {config.backend_store_uri}")
    logger.info(f"Default artifact root: {config.default_artifact_root}")
    logger.info("Press Ctrl+C to stop.")

    # Handle signals
    def handle_signal(signum, frame):
        print(f"Received signal {signum}, cleaning up.")
        cleanup(server_process)
        sys.exit()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return server_process


def main():

    parser = argparse.ArgumentParser(description="Start MLflow server.")
    parser.add_argument(
        "config",
        type=str,
        default="config.toml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    config = read_toml(args.config)

    server_process = start_mlflow_server(
        MLFlowServerConfig(**config["mlflow"]["server"])
    )
    server_process.wait()


if __name__ == "__main__":
    main()
