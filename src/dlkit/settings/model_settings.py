from collections.abc import Callable, Iterator

import torch.nn as nn
from lightning import LightningModule
from pydantic import Field, FilePath
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dlkit.datatypes.basic import FloatHyperparameter, IntHyperparameter
from .base_settings import HyperParameterSettings, ClassSettings
from dlkit.datatypes.dataset import Shape
from torchmetrics import Metric


class OptimizerSettings(HyperParameterSettings, ClassSettings[Optimizer]):
    name: str = Field(default="Adam", description="Optimizer name.")
    module_path: str = Field(default="torch.optim", description="Module path to the optimizer.")
    parameters: Iterator[nn.Parameter] | None = Field(
        default=None, description="Parameters to optimize.", frozen=False
    )
    lr: FloatHyperparameter | None = Field(default=None, description="Learning rate.")
    weight_decay: float = Field(default=0.0, description="Optional weight decay.")


class SchedulerSettings(ClassSettings[LRScheduler]):
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name.")
    module_path: str = Field(
        default="torch.optim.lr_scheduler", description="Module path to the scheduler."
    )
    factor: float = Field(default=0.5, description="Reduction factor.")
    patience: int = Field(
        default=50,
        description="Number of epochs with no improvement before reducing the LR.",
    )
    min_lr: float = Field(default=1e-8, description="Minimum learning rate.")


class TransformSettings(ClassSettings):
    name: str = Field(..., description="Name of the transform.")
    module_path: str = Field(
        default="dlkit.transforms", description="Module path to the transform."
    )
    dim: tuple[int, ...] | int = Field(
        default=0,
        description="List of dimensions to apply the transform on.",
    )


class MetricSettings(ClassSettings[Metric]):
    name: str = Field(default="MeanSquaredError", description="Name of the metric.")
    module_path: str = Field(
        default="torchmetrics.regression", description="Module path to the metric."
    )


class LossSettings(ClassSettings[Callable]):
    name: str = Field(default="mean_squared_error", description="Name of the loss.")
    module_path: str = Field(
        default="torchmetrics.functional.regression",
        description="Module path to the loss.",
    )


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

    loss_function: LossSettings = Field(
        default=LossSettings(),
        description="Loss function settings for training and validation.",
    )

    feature_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to features."
    )
    target_transforms: tuple[TransformSettings, ...] = Field(
        default=(), description="List of transforms to apply to targets."
    )

    is_autoencoder: bool = Field(default=False, description="Whether the model is an autoencoder.")

    metrics: tuple[MetricSettings, ...] = Field(
        default=(MetricSettings(),),
        description="List of metrics to compute on the model at test time.",
    )

    num_layers: IntHyperparameter | None = Field(default=None, description="Number of layers.")
    latent_size: IntHyperparameter | None = Field(
        default=None, description="Latent dimension size."
    )
    kernel_size: IntHyperparameter | None = Field(
        default=None, description="Convolution kernel size."
    )
    latent_channels: IntHyperparameter | None = Field(
        default=None, description="Number of latent channels before reduce to vector."
    )
    latent_width: IntHyperparameter | None = Field(
        default=None, description="Latent width before reduce to vector."
    )
    latent_height: IntHyperparameter | None = Field(
        default=None, description="Latent height before reduce to vector."
    )
