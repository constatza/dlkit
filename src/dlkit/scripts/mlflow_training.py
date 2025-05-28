import sys

import click
from dlkit.io.settings import load_validated_settings
from dlkit.run.mlflow_training import train_mlflow


@click.command(
    "MLFlow Training",
    help="Trains, tests, and predicts using the provided configuration.",
)
@click.argument("config-path")
def main(config_path: str = "./config.toml"):
    try:
        settings = load_validated_settings(config_path)
        train_mlflow(settings)
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
