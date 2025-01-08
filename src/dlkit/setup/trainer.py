from gc import callbacks

from lightning import Trainer
from dlkit.utils.system_utils import filter_kwargs
from pathlib import Path
from lightning.pytorch.callbacks import ModelSummary


def initialize_trainer(config):
    callbacks = [ModelSummary(max_depth=3)]
    total_params = {**config.get("trainer", {}), "default_root_dir": None}
    trainer = Trainer(**filter_kwargs(total_params), callbacks=callbacks)
    return trainer
