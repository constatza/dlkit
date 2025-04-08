import click
import sys
import atexit

from pydantic import FilePath
from loguru import logger

from dlkit.io.settings import load_validated_settings
from dlkit.setup.server import checks_before_start, start_server


@click.command("MLflow Server", help="Starts the MLflow server.")
@click.argument("config_path", default="./config.toml")
def server_cli(config_path: FilePath):
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
