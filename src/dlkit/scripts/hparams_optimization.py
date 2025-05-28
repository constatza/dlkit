import sys

import click

from dlkit.io.settings import load_validated_settings
from dlkit.run.optuna_training import train_optuna


@click.command("Hyperparameter Optimization", help="Hyperparameter Optimization with Optuna.")
@click.argument("config-path")
def main(config_path: str):
    """Main function to parse configuration and trigger training."""
    try:
        settings = load_validated_settings(config_path)
        train_optuna(settings)
    except KeyboardInterrupt:
        click.echo("Training interrupted by user.")
        sys.exit(1)
