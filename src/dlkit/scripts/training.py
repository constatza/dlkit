import sys

import click

from dlkit.io.settings import load_validated_settings
from dlkit.run.training import train


@click.command(
    "DLkit Training", help="Trains, tests, and predicts using the provided configuration."
)
@click.argument("config-path", type=str)
@click.option(
    "--optuna", is_flag=True, default=False, help="Use Optuna for hyperparameter optimization."
)
@click.option("--mlflow", is_flag=True, default=False, help="Use MLflow for experiment tracking.")
def main(config_path: str, optuna: bool, mlflow: bool) -> None:
    """Main function to parse configuration and trigger training."""
    try:
        settings = load_validated_settings(config_path)
        train(settings, optuna=optuna, mlflow=mlflow)
    except KeyboardInterrupt:
        click.echo("Training interrupted by user.")
        sys.exit(1)
