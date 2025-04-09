from dlkit.settings.general_settings import TrainerSettings, Settings
from dlkit.utils.system_utils import import_dynamic

from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
import mlflow
from pathlib import Path


def initialize_trainer(config: TrainerSettings) -> Trainer:

    callbacks = [
        ModelSummary(max_depth=3),
    ]
    for callback in config.callbacks:
        cb_class = import_dynamic(callback.name, prepend=callback.module_path)
        callbacks.append(cb_class(**callback.to_dict_compatible_with(cb_class)))

    trainer = Trainer(
        **config.to_dict_compatible_with(Trainer, exclude=("callbacks", "name")),
        callbacks=callbacks,
    )
    return trainer
