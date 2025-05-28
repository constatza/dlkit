import sys

import click
from loguru import logger

from dlkit.io.settings import load_validated_settings
from dlkit.run.training import train


@click.command("train", help="Trains, tests, and predicts using the provided configuration.")
@click.argument("config-path", type=str, default="./config.toml")
def main(config_path: str = "./config.toml") -> None:
    """Main function to parse configuration and trigger training."""
    try:
        settings = load_validated_settings(config_path)
        training_state = train(settings)
    except KeyboardInterrupt:
        logger.warning("Training interrupted.")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
