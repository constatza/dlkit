from typing import Any, Type
from collections.abc import Sequence
from typing import Self
import inspect

from pydantic import BaseModel
from optuna.trial import Trial
from optuna.distributions import CategoricalChoiceType

from dlkit.datatypes.basic import IntRange, FloatRange, IntHyper, FloatHyper, StrHyper


class BaseSettings(BaseModel):
    class Config:
        extra = "allow"
        validate_assignment = True
        frozen = True

    def to_dict_compatible_with(
        self, cls: Type, exclude: tuple[str, ...] = ()
    ) -> dict[str, Any]:
        signature = inspect.signature(cls)

        return {
            field: value
            for field, value in self.model_dump().items()
            if field in signature.parameters.keys() and not field in exclude
        }


class HyperParameterSettings(BaseSettings):

    def resolve(
        self,
        trial: Trial | None = None,
    ) -> Self:
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

        if isinstance(value, IntRange):
            return trial.suggest_int(
                name=field, low=value.low, high=value.high, step=value.step
            )
        if isinstance(value, FloatRange):
            return trial.suggest_float(
                name=field,
                low=value.low,
                high=value.high,
                step=value.step,
                log=value.log,
            )
        if isinstance(value, Sequence) and not isinstance(value, str):
            return trial.suggest_categorical(name=field, choices=value)

        return value
