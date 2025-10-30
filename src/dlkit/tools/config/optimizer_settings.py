from pydantic import Field
from pydantic_settings import SettingsConfigDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from dlkit.core.datatypes.base import FloatHyperparameter, StrHyperparameter
from .core.base_settings import ComponentSettings, HyperParameterSettings


class OptimizerSettings(ComponentSettings[Optimizer], HyperParameterSettings):
    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)
    name: StrHyperparameter = Field(default="AdamW", description="Optimizer name")
    module_path: str = Field(default="torch.optim", description="Module path to the optimizer")
    lr: FloatHyperparameter = Field(
        default=1e-3, description="Learning rate", alias="learning_rate"
    )
    weight_decay: FloatHyperparameter = Field(default=0.0, description="Optional weight decay")


class SchedulerSettings(ComponentSettings[LRScheduler]):
    model_config = SettingsConfigDict(extra="allow")
    name: str = Field(default="ReduceLROnPlateau", description="Scheduler name")
    module_path: str = Field(
        default="torch.optim.lr_scheduler", description="Module path to the scheduler"
    )
    factor: float = Field(default=0.5, description="Reduction factor")
    patience: int = Field(
        default=1000, description="Number of epochs with no improvement before reducing the LR"
    )
    min_lr: float = Field(default=1e-8, description="Minimum learning rate")
