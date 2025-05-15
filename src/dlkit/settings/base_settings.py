from collections.abc import Sequence
from typing import Any, Self, TypeVar

from optuna.distributions import CategoricalChoiceType
from optuna.trial import Trial
from pydantic import BaseModel, ConfigDict

from dlkit.datatypes.basic import FloatHyper, FloatRange, IntHyper, IntRange, StrHyper
from dlkit.utils.general import kwargs_compatible_with

T_co = TypeVar('T_co', covariant=True)


class BaseSettings(BaseModel):
	model_config = ConfigDict(
		frozen=True,
		validate_default=True,
		extra='allow',
	)

	def to_dict_compatible_with(
		self, cls: type, exclude: tuple[str, ...] = (), **kwargs
	) -> dict[str, Any]:
		return kwargs_compatible_with(cls, exclude=exclude, **kwargs, **self.model_dump())


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
			return trial.suggest_int(name=field, low=value.low, high=value.high, step=value.step)
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


class ClassSettings[T_co](BaseSettings):
	name: str
	module_path: str
	model_config = ConfigDict(
		arbitrary_types_allowed=True,
	)
