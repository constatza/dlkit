import sys
import traceback
import numpy as np

# from lightning.pytorch import seed_everything
from dlkit.io.readers import load_config, parse_config
from dlkit.io.logging import get_logger
from dlkit.setup.datamodule import initialize_datamodule
from dlkit.setup.trainer import initialize_trainer
from dlkit.setup.model import initialize_model
import torch
from typing import Dict

logger = get_logger(__name__)
torch.set_float32_matmul_precision("medium")
# seed_everything(1)


def train(config: Dict) -> None:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then executes the training, testing, and
    prediction steps. Finally, it saves the predictions to disk.

    Args:
        config (Dict): Configuration dictionary containing training parameters,
            paths, and model settings.
    """
    logger.info("Training started.")

    datamodule = initialize_datamodule(config)
    trainer = initialize_trainer(config)

    # Setup datamodule for training
    datamodule.setup(stage="fit")

    # Initialize model with shapes derived from datamodule
    model = initialize_model(config, datamodule.shapes)

    # Train and evaluate the model
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
    predictions = trainer.predict(model, datamodule=datamodule)

    # Convert predictions (list of Tensors) to a single NumPy array if possible
    if isinstance(predictions, list) and len(predictions) > 0:
        predictions_np = torch.cat(predictions, dim=0).numpy()
        np.save(config["paths"]["predictions"], predictions_np)

    logger.info("Training completed.")


def main() -> None:
    """Main function to parse configuration and trigger training."""
    try:
        config = parse_config(description="Training script.")
        train(config)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
