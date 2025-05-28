from typing import Literal
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from pydantic import Field

from .base_settings import BaseSettings
from .base_settings import ClassSettings


class PrunerSettings(ClassSettings[BasePruner]):
    name: str = Field(
        default="NopPruner",
        description="Pruner algorithm name for hyperparameter optimization.",
    )
    module_path: str = Field(default="optuna.pruners", description="Module path for the pruner.")
    n_warmup_steps: int | None = Field(
        default=None, description="Number of warmup steps before pruning starts."
    )
    interval_steps: int | None = Field(default=None, description="Interval between pruning steps.")


class SamplerSettings(ClassSettings[BaseSampler]):
    name: str = Field(
        default="TPESampler",
        description="Sampler algorithm name for hyperparameter optimization.",
    )
    module_path: str = Field(default="optuna.samplers", description="Module path for the sampler.")
    seed: int | None = Field(default=None, description="Optional random seed for reproducibility.")


class OptunaSettings(BaseSettings):
    enable: bool = Field(
        default=False, description="Whether to use Optuna for hyperparameter optimization."
    )
    n_trials: int = Field(
        default=10, description="Number of trials for hyperparameter optimization."
    )
    direction: Literal["minimize", "maximize"] = Field(
        default="minimize",
        description="Optimization direction.",
    )
    sampler: SamplerSettings = Field(default=SamplerSettings(), description="Optuna Sampler.")
    pruner: PrunerSettings = Field(default=PrunerSettings(), description="Optuna Pruner.")
