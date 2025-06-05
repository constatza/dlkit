from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelSummary, Callback

from dlkit.settings.general_settings import TrainerSettings
from dlkit.utils.loading import init_class
from loguru import logger as loguru_logger


def build_trainer(settings: TrainerSettings) -> Trainer:
    callbacks: list[Callback] = [ModelSummary(max_depth=2)]
    for callback in settings.callbacks:
        callbacks.append(init_class(callback))
        loguru_logger.info(f"Added callback: {callback.name}")

    if settings.logger.name:
        lightning_logger = init_class(settings.logger)
    else:
        lightning_logger = False

    trainer = Trainer(
        **settings.to_dict_compatible_with(Trainer, exclude=("callbacks", "name", "logger")),
        callbacks=callbacks,
        logger=lightning_logger,
        num_sanity_val_steps=0,
    )
    return trainer
