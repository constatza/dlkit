from lightning import Trainer
from dlkit.utils.system_utils import filter_kwargs
from pathlib import Path


def initialize_trainer(config):
    total_params = {**config.get("trainer", {}), "default_root_dir": None}
    trainer = Trainer(**filter_kwargs(total_params))
    return trainer
