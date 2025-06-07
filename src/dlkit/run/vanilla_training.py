from lightning.pytorch import LightningDataModule
from pydantic import validate_call, ConfigDict

from loguru import logger
from dlkit.settings import Settings
from dlkit.datatypes.learning import ModelState
from dlkit.setup.model_state import build_model_state


def train_state[M_T, D_T](training_state: ModelState[M_T, D_T]) -> ModelState[M_T, D_T]:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then executes the training, testing, and
    prediction steps. Finally, it saves the predictions to disk.

    Args:
        training_state: The training state containing the model, datamodule,
            and trainer to be used for training.
    Returns:
        ModelState: An object containing the trained model, datamodule,
            and trainer after training, testing, and prediction.
    """

    # Initialize datamodule and trainer

    # Initialize model with shapes derived from datamodule
    model = training_state.model
    datamodule = training_state.datamodule
    trainer = training_state.trainer

    # Train and evaluate the model
    trainer.fit(model, datamodule=datamodule)
    return ModelState(trainer=trainer, model=model, datamodule=datamodule)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_vanilla(
    settings: Settings,
    datamodule: LightningDataModule | None = None,
) -> ModelState:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the training state and executes the training,
    testing, and prediction steps. It returns the final training state.

    Args:
        settings: The configuration object for the training process.
        datamodule: An optional datamodule to use. If not provided, it will be
            built from the settings.
    """
    training_state = build_model_state(settings, datamodule=datamodule)

    if settings.MODEL.train:
        logger.info("Starting training.")
        training_state = train_state(training_state)
        logger.info("Training completed.")
    else:
        logger.info("Training skipped as per configuration.")

    trainer = training_state.trainer
    if settings.MODEL.predict:
        trainer.predict(training_state.model, datamodule=training_state.datamodule)
        logger.info("Prediction completed.")
    if settings.MODEL.test:
        trainer.test(training_state.model, datamodule=training_state.datamodule)
        logger.info("Testing completed.")
    return training_state
