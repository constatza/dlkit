from collections.abc import Callable, Iterator

import torch.nn as nn
from lightning import LightningModule
from pydantic import Field, field_validator, FilePath
from pydantic_core.core_schema import ValidationInfo
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dlkit.datatypes.basic import FloatHyper, IntHyper
from .base_settings import HyperParameterSettings, ClassSettings
from dlkit.datatypes.dataset import Shape


class OptimizerSettings(HyperParameterSettings, ClassSettings[Optimizer]):
    name: str = Field(default="Adam", description="Optimizer name.")
    module_path: str = Field(default="torch.optim", description="Module path to the optimizer.")
    parameters: Iterator[nn.Parameter] | None = Field(
        default=None, description="Parameters to optimize.", frozen=False
    )
    lr: FloatHyper | None = Field(default=None, description="Learning rate.")
    weight_decay: float = Field(default=0.0, description="Optional weight decay.")


class SchedulerSettings(ClassSettings[LRScheduler]):
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name.")
    module_path: str = Field(
        default="torch.optim.lr_scheduler", description="Module path to the scheduler."
    )
    factor: float = Field(default=0.8, description="Reduction factor.")
    patience: int = Field(
        default=20,
        description="Number of epochs with no improvement before reducing the LR.",
    )
    min_lr: float = Field(default=1e-5, description="Minimum learning rate.")


class TransformSettings(ClassSettings):
    name: str = Field(..., description="Name of the transform.")
    module_path: str = Field(
        default="dlkit.transforms", description="Module path to the transform."
    )
    dim: tuple[int, ...] | int = Field(
        default=0,
        description="List of dimensions to apply the transform on.",
    )


class LossFunctionSettings(ClassSettings[Callable]):
    name: str = Field(default="MSELoss", description="Name of the loss function.")
    module_path: str = Field(default="torch.nn", description="Module path to the loss function.")


class ModelSettings(HyperParameterSettings, ClassSettings[LightningModule]):
    name: str = Field(..., description="Model namespace path.")
    module_path: str = Field(
        default="dlkit.networks",
        description="Module path to the model.",
    )

    shape: Shape | None = Field(default=None, description="Input shape of the model.")

    optimizer: OptimizerSettings = Field(
        default=OptimizerSettings(), description="Optimizer settings."
    )
    scheduler: SchedulerSettings = Field(
        default=SchedulerSettings(), description="Scheduler settings."
    )

    checkpoint: FilePath | None = Field(default=None, description="Path to the model checkpoint.")

    train: bool = Field(default=True, description="Whether to train the model.")

    test: bool = Field(default=True, description="Whether to test the model.")

    predict: bool = Field(default=True, description="Whether to predict with the model.")

    train_loss: LossFunctionSettings = Field(
        default=LossFunctionSettings(),
        description="Loss function settings for training and validation.",
    )
    test_loss: LossFunctionSettings | None = Field(
        default=None, description="Loss function settings for testing."
    )

    feature_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to features."
    )
    target_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to targets."
    )

    is_autoencoder: bool = Field(default=False, description="Whether the model is an autoencoder.")

    num_layers: IntHyper | None = Field(default=None, description="Number of layers.")
    latent_size: IntHyper | None = Field(default=None, description="Latent dimension size.")
    kernel_size: IntHyper | None = Field(default=None, description="Convolution kernel size.")
    latent_channels: IntHyper | None = Field(
        default=None, description="Number of latent channels before reduce to vector."
    )
    latent_width: IntHyper | None = Field(
        default=None, description="Latent width before reduce to vector."
    )
    latent_height: IntHyper | None = Field(
        default=None, description="Latent height before reduce to vector."
    )

    @field_validator("test_loss")
    @classmethod
    def populate_test_loss(cls, value, info: ValidationInfo):
        if not value:
            return info.data["train_loss"]
        return value
