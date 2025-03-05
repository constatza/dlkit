from cgitb import enable

from lightning import Trainer
from dlkit.utils.system_utils import filter_kwargs
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
from pathlib import Path


def initialize_trainer(config):

    callbacks = [
        ModelSummary(max_depth=3),
    ]
    if config.get("mlflow").get("enable_checkpointing", False):
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(mlflow.get_artifact_uri()) / "checkpoints",
                monitor="val_loss",  # Metric to monitor
                mode="min",  # "min" for loss, "max" for accuracy or other metrics
                save_top_k=1,  # Save only the best model
                filename="best-{epoch:02d}",  # Naming pattern
                verbose=True,  # Logs when a checkpoint is saved
                every_n_epochs=10,  # Save checkpoint every n epochs
            ),
        )

    total_params = {
        **config.get("trainer", {}),
        "default_root_dir": None,
        "logger": False,
    }
    trainer = Trainer(
        **filter_kwargs(total_params),
        callbacks=callbacks,
    )
    return trainer
