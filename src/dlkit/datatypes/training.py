import attr

from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from dlkit.datamodules import NumpyModule


@attr.frozen
class TrainingState:
    trainer: Trainer = attr.field()
    model: LightningModule = attr.field()
    datamodule: NumpyModule = attr.field()
