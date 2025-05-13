from attr import define, field
from lightning.pytorch import LightningModule, Trainer, LightningDataModule


@define(frozen=True)
class TrainingState:
	trainer: Trainer = field()
	model: LightningModule = field()
	datamodule: LightningDataModule = field()
