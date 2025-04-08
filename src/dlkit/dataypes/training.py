import attr
from lightning.pytorch import Trainer, LightningModule, LightningDataModule
import numpy as np


@attr.frozen
class TrainingState:
    trainer: Trainer = attr.field()
    model: LightningModule = attr.field()
    datamodule: LightningDataModule = attr.field()
    predictions: np.ndarray | None = attr.field(default=None)
