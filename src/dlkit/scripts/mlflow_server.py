import atexit
import sys

import click
from loguru import logger

from dlkit.io.settings import load_validated_settings
from dlkit.setup.server import start_server


@click.command("MLflow Server", help="Starts the MLflow server.")
@click.argument("config_path", default="./config.toml")
def server_cli(config_path: str = "./config.toml"):
    """
    Command-line interface to start the MLflow server.

    This function initializes the MLflow server using the specified configuration
    file. It validates the configuration, starts the server, and ensures that
    the server process is terminated upon exit.

    Args:
        config_path (str): Path to the configuration file, default is './config.toml'.
    """
    settings = load_validated_settings(config_path)
    server = start_server(settings.MLFLOW.server)
    atexit.register(lambda: server.terminate())
    server.wait()


def main():
    try:
        server_cli()
    except Exception as e:
        logger.error(e)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    server_cli()
