import torch
from loguru import logger
from dlkit.datatypes.learning import ModelState
from lightning.pytorch import LightningDataModule, seed_everything
from dlkit.settings import Settings
from dlkit.setup.datamodule import build_datamodule
from dlkit.setup.model import build_model
from dlkit.setup.trainer import build_trainer


def build_model_state(
    settings: Settings, datamodule: LightningDataModule | None = None
) -> ModelState:
    """Builds the training state based on the provided settings.

    This function initializes the datamodule, trainer, and model using the
    provided configuration. It then returns a TrainingState object containing
    these components.

    Args:
        settings: The configuration object for the training process.
        datamodule: An optional datamodule to use. If not provided, it will be


    Returns:
        ModelState: An object containing the initialized trainer, model,
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
        model = model.__class__.load_from_checkpoint(ckpt_path=ckpt, strict=False)

    return ModelState(trainer=trainer, model=model, datamodule=datamodule, settings=settings)
