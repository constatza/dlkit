from dlkit.settings.general_settings import TrainerSettings, Settings

from lightning import Trainer
from lightning.pytorch.callbacks import ModelSummary, ModelCheckpoint
import mlflow
from pathlib import Path


def initialize_trainer(config: TrainerSettings) -> Trainer:

    callbacks = [
        ModelSummary(max_depth=3),
    ]
    if config.enable_checkpointing:
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

    trainer = Trainer(
        **config.to_dict_compatible_with(Trainer),
        callbacks=callbacks,
    )
    return trainer
