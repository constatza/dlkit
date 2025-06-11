from typing import Any, Self

from optuna.distributions import CategoricalChoiceType
from optuna.trial import Trial
from pydantic import BaseModel, ConfigDict, validate_call

from dlkit.datatypes.basic import (
    Hyperparameter,
    IntDistribution,
    FloatDistribution,
    BoolDistribution,
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
    @validate_call(config=ConfigDict(arbitrary_types_allowed=True))
    def get_optuna_suggestion(trial: Trial, field: str, value: Hyperparameter | Any) -> Any:
        """Get an Optuna suggestion based on the type of hyperparameter.

        Args:
            trial (Trial): The Optuna trial object.
            field (str): The name of the hyperparameter field.
            value (IntHyper | FloatHyper | StrHyper): The hyperparameter value to suggest.
        Returns:
            CategoricalChoiceType: The suggested value for the hyperparameter.
        """
        if isinstance(value, CategoricalDistribution) or isinstance(value, BoolDistribution):
            return trial.suggest_categorical(field, choices=value.choices)
        if isinstance(value, IntDistribution):
            return trial.suggest_int(field, low=value.low, high=value.high, step=value.step)
        if isinstance(value, FloatDistribution):
            return trial.suggest_float(
                field, low=value.low, high=value.high, step=value.step, log=value.log
            )
        return value
