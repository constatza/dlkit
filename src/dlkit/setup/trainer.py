from gc import callbacks

from lightning import Trainer
from dlkit.utils.system_utils import filter_kwargs
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint


def initialize_trainer(config):

    callbacks = [
        ModelSummary(max_depth=3),
        ModelCheckpoint(
            monitor="val_loss",  # Metric to monitor
            mode="min",  # "min" for loss, "max" for accuracy or other metrics
            save_top_k=1,  # Save only the best model
            filename="best_model-{epoch:02d}-{val_loss:.2f}",  # Naming pattern
            verbose=True,  # Logs when a checkpoint is saved
        ),
    ]
    total_params = {**config.get("trainer", {}), "default_root_dir": None}
    total_params["logger"] = False
    trainer = Trainer(**filter_kwargs(total_params), callbacks=callbacks)
    return trainer
