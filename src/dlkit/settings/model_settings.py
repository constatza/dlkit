from pydantic import Field

from .types import IntHyper, FloatHyper, Shape
from .base_settings import BaseSettings, HyperParameterSettings


class OptimizerSettings(HyperParameterSettings):
    name: str = Field(default="Adam", description="Optimizer name.")
    lr: FloatHyper | None = Field(default=None, description="Learning rate.")
    weight_decay: float | None = Field(
        default=None, description="Optional weight decay."
    )


class SchedulerSettings(BaseSettings):
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name.")
    factor: float = Field(default=0.8, description="Reduction factor.")
    patience: int = Field(
        default=10,
        description="Number of epochs with no improvement before reducing the LR.",
    )
    min_lr: float = Field(default=1e-5, description="Minimum learning rate.")


class ModelSettings(HyperParameterSettings):
    class Config:
        arbitrary_types_allowed = True

    name: str = Field(..., description="Model namespace path.")
    shape: Shape = Field(default=Shape(), description="Model shape.")
    optimizer: OptimizerSettings = Field(
        default=OptimizerSettings(), description="Optimizer settings."
    )
    scheduler: SchedulerSettings = Field(
        default=SchedulerSettings(), description="Scheduler settings."
    )
    num_layers: IntHyper | None = Field(default=None, description="Number of layers.")
    latent_size: IntHyper | None = Field(
        default=None, description="Latent dimension size."
    )
    kernel_size: IntHyper | None = Field(
        default=None, description="Convolution kernel size."
    )
    latent_channels: IntHyper | None = Field(
        default=None, description="Number of latent channels before reduce to vector."
    )
    latent_width: IntHyper | None = Field(
        default=None, description="Latent width before reduce to vector."
    )
    latent_height: IntHyper | None = Field(
        default=None, description="Latent height before reduce to vector."
    )
