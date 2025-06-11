"""A module defining the ModelState class for managing training sessions in PyTorch Lightning."""

from typing import TypeVar
from attr import define, field
from lightning.pytorch import Trainer, LightningModule, LightningDataModule
from dlkit.settings.general_settings import Settings

Model_T = TypeVar("Model_T", bound=LightningModule)
DataModule_T = TypeVar("DataModule_T", bound=LightningDataModule)


@define(frozen=True)
class ModelState[Model_T, DataModule_T]:
    """A class to hold the state of a training session.
    It includes the trainer, model, and datamodule used during training.

    Args:
        trainer: The Trainer instance used for training.
        model: The model being trained.
        datamodule: The DataModule instance providing the data.
    """

    model: Model_T = field(kw_only=True)
    trainer: Trainer | None = field(kw_only=True)
    datamodule: DataModule_T = field(kw_only=True)
    settings: Settings = field(default=None, kw_only=True)
