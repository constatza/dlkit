import sys
import traceback

import click
from loguru import logger

from dlkit.io.settings import load_validated_settings
from dlkit.run.optuna_training import train_optuna


@click.command("Hyperparameter Optimization", help="Hyperparameter Optimization with Optuna.")
@click.argument("config-path")
def main(config_path: str = "./config.toml"):
    """Main function to parse configuration and trigger training."""
    try:
        settings = load_validated_settings(config_path)
        train_optuna(settings)
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
