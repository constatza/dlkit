import sys
import traceback

import click
import torch
from lightning.pytorch import seed_everything
from loguru import logger
from pydantic import validate_call

from dlkit.datatypes.learning import TrainingState
from dlkit.io.settings import Settings, load_validated_settings
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.model import initialize_model
from dlkit.setup.trainer import initialize_trainer

torch.set_float32_matmul_precision("medium")
seed_everything(1)


@validate_call
def train(settings: Settings) -> TrainingState:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then executes the training, testing, and
    prediction steps. Finally, it saves the predictions to disk.

    Args:
        config_path (FilePath): The path to the configuration file.
    """
    logger.info("Training started.")

    datamodule = initialize_datamodule(
        settings.DATA, settings.PATHS, datamodule_device=settings.TRAINER.accelerator
    )
    trainer = initialize_trainer(settings.TRAINER)

    # Initialize model with shapes derived from datamodule
    datamodule.setup(stage="fit")
    model = initialize_model(settings.MODEL, datamodule.shape)

    # Train and evaluate the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    trainer.predict(model, datamodule=datamodule)

    logger.info("Training completed.")
    return TrainingState(trainer=trainer, model=model, datamodule=datamodule)


@click.command(
    "train", help="Trains, tests, and predicts using the provided configuration."
)
@click.argument("config-path")
def train_cli(config_path: str = "./config.toml"):
    settings = load_validated_settings(config_path)
    training_state = train(settings)
    return training_state


def main() -> None:
    """Main function to parse configuration and trigger training."""
    try:
        train_cli()
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
