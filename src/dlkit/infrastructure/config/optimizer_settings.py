from collections.abc import Callable
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict

from .core.base_settings import (
    ComponentSettings,
    HyperParameterSettings,
    validate_module_path_import,
)
from .core.types import FloatHyperparameter, PositiveFloatHyperparameter


class OptimizerSettings(ComponentSettings, HyperParameterSettings):
    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)
    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="AdamW",
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Optimizer name",
    )
    module_path: str | None = Field(
        default="torch.optim",
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Module path to the optimizer",
    )
    lr: PositiveFloatHyperparameter = Field(
        default=1e-3, description="Learning rate", alias="learning_rate"
    )
    weight_decay: FloatHyperparameter = Field(default=0.0, description="Optional weight decay")

    @field_validator("module_path", mode="after")
    @classmethod
    def _validate_module_path(cls, v: str | None) -> str | None:
        return validate_module_path_import(v)


class SchedulerSettings(ComponentSettings):
    model_config = SettingsConfigDict(extra="allow")
    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="ReduceLROnPlateau",
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Scheduler name",
    )
    module_path: str | None = Field(
        default="torch.optim.lr_scheduler",
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Module path to the scheduler",
    )
    factor: float = Field(default=0.5, description="Reduction factor")
    patience: int = Field(
        default=1000, description="Number of epochs with no improvement before reducing the LR"
    )
    min_lr: float = Field(default=1e-8, description="Minimum learning rate")
