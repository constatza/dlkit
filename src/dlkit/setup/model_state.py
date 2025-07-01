import torch

from pydantic import FilePath
from lightning.pytorch import LightningDataModule, seed_everything
from loguru import logger
from dlkit.datatypes.run import ModelState
from dlkit.settings import Settings, RunMode
from dlkit.setup.datamodule import build_datamodule
from dlkit.setup.model import build_model
from dlkit.setup.trainer import build_trainer
from dlkit.utils.loading import init_class


def build_model_state(
    settings: Settings,
    datamodule: LightningDataModule | None = None,
    checkpoint: FilePath | None = None,
) -> ModelState:
    """Builds the training state based on the provided settings.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then returns a TrainingState object containing
    these components.

    Args:
        settings: The configuration object for the training process.
        datamodule: An optional datamodule to use. If not provided, it will be
            built using the settings.
        checkpoint: An optional checkpoint path to load the model from.

    Returns:
        ModelState: An object containing the initialized trainer, model,
        and datamodule.
    """
    torch.set_float32_matmul_precision(settings.RUN.precision)
    seed_everything(settings.RUN.seed)
    trainer = (
        build_trainer(settings.TRAINER) if settings.RUN.mode is not RunMode.INFERENCE else None
    )

    dataset = init_class(settings.DATASET)

    datamodule = datamodule or build_datamodule(
        settings=settings.DATAMODULE,
        dataset=dataset,
        dataloader_settings=settings.DATALOADER,
        paths=settings.PATHS,
    )

    model = build_model(settings=settings.MODEL, dataset=dataset)
    checkpoint = checkpoint or settings.MODEL.checkpoint
    if checkpoint and (settings.RUN.mode is RunMode.INFERENCE):
        logger.info(f"Loading model from checkpoint: {checkpoint}")
        model = model.__class__.load_from_checkpoint(checkpoint)

    return ModelState(trainer=trainer, model=model, datamodule=datamodule, settings=settings)
