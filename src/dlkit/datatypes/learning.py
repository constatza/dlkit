"""A module defining the TrainingState class for managing training sessions in PyTorch Lightning."""

from attr import define, field
from lightning.pytorch import Trainer


@define(frozen=True)
class TrainingState[Model_T, DataModule_T]:
    """A class to hold the state of a training session.
    It includes the trainer, model, and datamodule used during training.

    Args:
        trainer: The Trainer instance used for training.
        model: The model being trained.
        datamodule: The DataModule instance providing the data.
    """

    trainer: Trainer = field()
    model: Model_T = field()
    datamodule: DataModule_T = field()
    precision: str = field(default="medium", kw_only=True)
    seed: int = field(default=1, kw_only=True)
