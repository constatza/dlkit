"""Search settings — hyperparameter optimization configuration."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field
from pydantic.types import PositiveInt

from dlkit.infrastructure.config.core.base_settings import BasicSettings


class FloatParam(BasicSettings):
    """Uniform float hyperparameter range.

    Args:
        type: Discriminator tag, must be "float".
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    type: Literal["float"]
    low: float
    high: float


class LogFloatParam(BasicSettings):
    """Log-uniform float hyperparameter range.

    Args:
        type: Discriminator tag, must be "log_float".
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    type: Literal["log_float"]
    low: float
    high: float


class IntParam(BasicSettings):
    """Integer hyperparameter range.

    Args:
        type: Discriminator tag, must be "int".
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
        step: Optional step size between values.
    """

    type: Literal["int"]
    low: int
    high: int
    step: int | None = None


class LogIntParam(BasicSettings):
    """Log-uniform integer hyperparameter range.

    Args:
        type: Discriminator tag, must be "log_int".
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """

    type: Literal["log_int"]
    low: int
    high: int


class CategoricalParam(BasicSettings):
    """Categorical hyperparameter choices.

    Args:
        type: Discriminator tag, must be "categorical".
        choices: List of possible values.
    """

    type: Literal["categorical"]
    choices: list[Any]


class ConstantParam(BasicSettings):
    """Fixed constant hyperparameter (no search).

    Args:
        type: Discriminator tag, must be "constant".
        value: The fixed value.
    """

    type: Literal["constant"]
    value: Any


SpaceParam = Annotated[
    FloatParam | LogFloatParam | IntParam | LogIntParam | CategoricalParam | ConstantParam,
    Field(discriminator="type"),
]


class SamplerSettings(BasicSettings):
    """1-1 mapping to optuna.samplers.* __init__ — field names must not change.

    Args:
        name: Sampler class name.
        module_path: Python module path for the sampler.
        seed: Optional random seed for reproducibility.
    """

    name: str | None = "TPESampler"
    module_path: str | None = "optuna.samplers"
    seed: int | None = None


class PrunerSettings(BasicSettings):
    """1-1 mapping to optuna.pruners.* __init__ — field names must not change.

    Args:
        name: Pruner class name.
        module_path: Python module path for the pruner.
        n_warmup_steps: Number of warmup steps before pruning starts.
        interval_steps: Interval between pruning steps.
    """

    name: str | None = "NopPruner"
    module_path: str | None = "optuna.pruners"
    n_warmup_steps: int | None = None
    interval_steps: int | None = None


class SearchSettings(BasicSettings):
    """HPO configuration. search.sampler and search.pruner map 1-1 to Optuna classes.

    Args:
        n_trials: Number of optimization trials.
        direction: Optimization direction (minimize or maximize).
        objective: Metric name used as the HPO objective.
        study_name: Optional Optuna study name for persistence.
        storage: Optional Optuna storage URL.
        sampler: Sampler configuration.
        pruner: Pruner configuration.
        space: Hyperparameter search space keyed by dotted config path.
    """

    n_trials: PositiveInt = 10
    direction: Literal["minimize", "maximize"] = "minimize"
    objective: str = "val/loss"
    study_name: str | None = None
    storage: str | None = None
    sampler: SamplerSettings = Field(default_factory=SamplerSettings)
    pruner: PrunerSettings = Field(default_factory=PrunerSettings)
    space: dict[str, SpaceParam] = Field(default_factory=dict)
