import torch
from lightning.pytorch import LightningDataModule, seed_everything
from pydantic import validate_call, ConfigDict

from loguru import logger
from dlkit.setup.datamodule import build_datamodule
from dlkit.setup.model import build_model
from dlkit.setup.trainer import build_trainer
from dlkit.settings import Settings
from dlkit.datatypes.learning import TrainingState


def build_training_state(
    settings: Settings, datamodule: LightningDataModule | None = None
) -> TrainingState:
    """Builds the training state based on the provided settings.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then returns a TrainingState object containing
    these components.

    Args:
        settings: The configuration object for the training process.
        datamodule: An optional datamodule to use. If not provided, it will be


    Returns:
        TrainingState: An object containing the initialized trainer, model,
        and datamodule.
    """
    torch.set_float32_matmul_precision(settings.precision)
    seed_everything(settings.seed)

    trainer = build_trainer(settings.TRAINER)

    if datamodule is None:
        datamodule = build_datamodule(
            settings.DATAMODULE,
            settings.DATASET,
            settings.DATALOADER,
            settings.PATHS,
        )

    model = build_model(
        settings=settings.MODEL,
        settings_path=settings.PATHS.settings,
        dataset=datamodule.dataset.raw,
    )
    if ckpt := settings.MODEL.checkpoint:
        logger.info(f"Loading model from checkpoint: {ckpt}")
        model = model.__class__.load_from_checkpoint(checkpoint_path=ckpt, strict=False)

    return TrainingState(trainer=trainer, model=model, datamodule=datamodule, seed=settings.seed)


def train_state[M_T, D_T](training_state: TrainingState[M_T, D_T]) -> TrainingState[M_T, D_T]:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then executes the training, testing, and
    prediction steps. Finally, it saves the predictions to disk.

    Args:
        training_state: The training state containing the model, datamodule,
            and trainer to be used for training.
    Returns:
        TrainingState: An object containing the trained model, datamodule,
            and trainer after training, testing, and prediction.
    """

    # Initialize datamodule and trainer

    # Initialize model with shapes derived from datamodule
    model = training_state.model
    datamodule = training_state.datamodule
    trainer = training_state.trainer

    # Train and evaluate the model
    trainer.fit(model, datamodule=datamodule)
    return TrainingState[type(model), type(datamodule)](
        trainer=trainer, model=model, datamodule=datamodule
    )


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def train_vanilla(
    settings: Settings,
    datamodule: LightningDataModule | None = None,
) -> TrainingState:
    """Trains, tests, and predicts using the provided configuration.

    This function initializes the training state and executes the training,
    testing, and prediction steps. It returns the final training state.

    Args:
        settings: The configuration object for the training process.
        datamodule: An optional datamodule to use. If not provided, it will be
            built from the settings.
        predict: Whether to perform prediction after training and testing.
    """
    training_state = build_training_state(settings, datamodule=datamodule)

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
