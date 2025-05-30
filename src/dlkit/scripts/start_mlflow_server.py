import sys

import click
from loguru import logger

from dlkit.io.settings import load_validated_settings
from dlkit.setup.mlflow_server import ServerProcess


@click.command("MLflow Server", help="Starts the MLflow server.")
@click.argument("config_path", default="./config.toml")
def main(config_path: str = "./config.toml"):
    """Command-line interface to start the MLflow server.

    This function initializes the MLflow server using the specified configuration
    file. It validates the configuration, starts the server, and ensures that
    the server process is terminated upon exit.

    Args:
        config_path (str): Path to the configuration file, default is './config.toml'.
    """
    settings = load_validated_settings(config_path)
    with ServerProcess(settings.MLFLOW.server) as server:
        try:
            server.process.wait()
        except KeyboardInterrupt:
            logger.info("MLflow server stopped by user.")
        except Exception as e:
            logger.error(f"Error while running MLflow server: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
