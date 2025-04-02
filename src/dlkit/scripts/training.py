import sys
import traceback
import numpy as np
import torch
import click

from lightning.pytorch import seed_everything
from pydantic import validate_call, FilePath
from loguru import logger

from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
from dlkit.io.settings import load_validated_settings

torch.set_float32_matmul_precision("medium")
seed_everything(1)


@click.command(
    "train", help="Trains, tests, and predicts using the provided configuration."
)
@click.argument("config-path")
@validate_call
def train(config_path: FilePath) -> None:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then executes the training, testing, and
    prediction steps. Finally, it saves the predictions to disk.

    Args:
        config_path (FilePath): The path to the configuration file.
    """
    logger.info("Training started.")
    settings = load_validated_settings(config_path)

    datamodule = initialize_datamodule(settings.DATAMODULE, settings.PATHS)
    trainer = initialize_trainer(settings.TRAINER)

    # Initialize model with shapes derived from datamodule
    datamodule.setup(stage="fit")
    model = initialize_model(settings.MODEL, datamodule.shape)

    # Train and evaluate the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    predictions = trainer.predict(model, datamodule=datamodule)

    # Convert predictions (list of Tensors) to a single NumPy array if possible
    if isinstance(predictions, list) and len(predictions) > 0:
        predictions_np = torch.cat(predictions, dim=0).numpy()
        np.save(str(settings.PATHS.predictions), predictions_np)

    logger.info("Training completed.")


def main() -> None:
    """Main function to parse configuration and trigger training."""
    try:
        train()
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
