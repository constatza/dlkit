from typing import Any, Self

from optuna.distributions import CategoricalChoiceType
from optuna.trial import Trial
from pydantic import BaseModel, ConfigDict

from dlkit.datatypes.basic import (
    FloatHyper,
    FloatDistribution,
    IntHyper,
    IntDistribution,
    StrHyper,
    CategoricalDistribution,
)
from dlkit.utils.general import kwargs_compatible_with


class BaseSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        validate_default=True,
        extra="allow",
    )

    def to_dict_compatible_with(
        self, cls: type, exclude: tuple[str, ...] = (), **kwargs
    ) -> dict[str, Any]:
        return kwargs_compatible_with(cls, exclude=exclude, **kwargs, **self.model_dump())


class ClassSettings[T](BaseSettings):
    """Settings that include a class name and module path for dynamic loading.

    Attributes:
            name (str): The name of the class to be loaded.
            module_path (str): The module path where the class is defined.
    """

    name: str
    module_path: str
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class HyperParameterSettings(BaseSettings):
    def resolve(
        self,
        trial: Trial | None = None,
    ) -> Self:
        """Resolve hyperparameters using Optuna trial suggestions.
        If no trial is provided, returns a copy of the current model without modifications.

        Args:
            trial (Trial | None): The Optuna trial object to use for suggestions. If None, returns a copy of the model.
        Returns:
            Self: A new instance of the model with resolved hyperparameters.
        """
        if trial is None:
            return self.model_copy()

        resolved: dict[str, CategoricalChoiceType] = {}
        for field in self.model_fields_set:
            value = getattr(self, field)
            resolved[field] = self.get_optuna_suggestion(trial, field, value)

        return self.model_copy(update=resolved)

    @staticmethod
    def get_optuna_suggestion(
        trial: Trial, field: str, value: IntHyper | FloatHyper | StrHyper
    ) -> CategoricalChoiceType:
        """Get an Optuna suggestion based on the type of hyperparameter.

        Args:
            trial (Trial): The Optuna trial object.
            field (str): The name of the hyperparameter field.
            value (IntHyper | FloatHyper | StrHyper): The hyperparameter value to suggest.
        Returns:
            CategoricalChoiceType: The suggested value for the hyperparameter.
        """
        if isinstance(value, IntDistribution):
            return trial.suggest_int(name=field, low=value.low, high=value.high, step=value.step)
        if isinstance(value, FloatDistribution):
            return trial.suggest_float(
                name=field,
                low=value.low,
                high=value.high,
                step=value.step,
                log=value.log,
            )
        if isinstance(value, CategoricalDistribution):
            return trial.suggest_categorical(name=field, choices=value.choices)
        return value
