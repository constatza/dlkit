import sys
from typing import Literal

import click
from dlkit.run import run_from_path


@click.command(
    "DLkit Training", help="Trains, tests, and predicts using the provided configuration."
)
@click.argument("config-path", type=str)
@click.option(
    "--mode", type=click.Choice(["training", "inference", "mlflow", "optuna"]), default="training"
)
def main(config_path: str, mode: Literal["training", "inference", "mlflow", "optuna"]) -> None:
    """Main function to parse configuration and trigger training."""
    try:
        run_from_path(config_path, mode=mode)
    except KeyboardInterrupt:
        click.echo("Training interrupted by user.")
        sys.exit(1)
