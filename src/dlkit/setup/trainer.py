from pathlib import Path

import mlflow
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from dlkit.settings.general_settings import Settings, TrainerSettings
from dlkit.utils.system_utils import import_dynamic


def initialize_trainer(config: TrainerSettings) -> Trainer:

    callbacks = []
    for callback in config.callbacks:
        cb_class = import_dynamic(callback.name, prepend=callback.module_path)
        callbacks.append(cb_class(**callback.to_dict_compatible_with(cb_class)))

    if config.logger.name is not None:
        logger_class = import_dynamic(
            config.logger.name, prepend=config.logger.module_path
        )
        logger = logger_class(**config.logger.to_dict_compatible_with(logger_class))
    else:
        logger = False

    trainer = Trainer(
        **config.to_dict_compatible_with(
            Trainer, exclude=("callbacks", "name", "logger")
        ),
        callbacks=callbacks,
        logger=logger,
    )
    return trainer
