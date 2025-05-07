from attr import define, field
from lightning.pytorch import LightningModule, Trainer
from dlkit.datamodules import InMemoryModule


@define(frozen=True)
class TrainingState:
	trainer: Trainer = field()
	model: LightningModule = field()
	datamodule: InMemoryModule = field()
