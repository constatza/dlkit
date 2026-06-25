"""Concrete hyperparameter applicator implementations.

These applicators implement the IHyperparameterApplicator protocol
to apply sampled hyperparameters to workflow settings.
"""

from __future__ import annotations

from typing import Any

from dlkit.infrastructure.config.job_config import JobConfig
from dlkit.infrastructure.config.search_settings import (
    CategoricalParam,
    ConstantParam,
    FloatParam,
    IntParam,
    LogFloatParam,
    LogIntParam,
    SpaceParam,
)


def suggest_from_space(
    trial: Any,  # ponytail: Optuna Trial has no public Protocol; Any is the boundary
    space: dict[str, SpaceParam],
) -> dict[str, str | int | float | bool | None]:
    """Sample hyperparameter values from a typed search space.

    Args:
        trial: An Optuna trial object.
        space: Typed search space keyed by dotted config path.

    Returns:
        Mapping of dotted config path → sampled scalar value.
    """
    suggestions: dict[str, str | int | float | bool | None] = {}
    for dotpath, param in space.items():
        match param:
            case FloatParam(low=low, high=high):
                suggestions[dotpath] = trial.suggest_float(dotpath, low, high)
            case LogFloatParam(low=low, high=high):
                suggestions[dotpath] = trial.suggest_float(dotpath, low, high, log=True)
            case IntParam(low=low, high=high, step=step):
                suggestions[dotpath] = trial.suggest_int(dotpath, low, high, step=step or 1)
            case LogIntParam(low=low, high=high):
                suggestions[dotpath] = trial.suggest_int(dotpath, low, high, log=True)
            case CategoricalParam(choices=choices):
                suggestions[dotpath] = trial.suggest_categorical(dotpath, list(choices))
            case ConstantParam(value=value):
                suggestions[dotpath] = value
    return suggestions


class ModelSettingsApplicator:
    """Applies sampled hyperparameters to the model section of JobConfig.

    Patches the settings via dotted-path keys so that nested model
    hyperparameters are updated in a single immutable copy operation.
    """

    def apply(
        self,
        base_settings: JobConfig,
        hyperparameters: dict[str, str | int | float | bool | None],
    ) -> JobConfig:
        """Apply hyperparameters to settings via dotted-path patch.

        Args:
            base_settings: A JobConfig instance to patch.
            hyperparameters: Sampled hyperparameters keyed by dotted config path.

        Returns:
            Settings with hyperparameters applied, or original when empty.
        """
        if not hyperparameters:
            return base_settings
        return base_settings.patch(hyperparameters)
