import inspect
from collections.abc import Sequence
from typing import Any, Self

from optuna.distributions import CategoricalChoiceType
from optuna.trial import Trial
from pydantic import BaseModel, ConfigDict, FilePath

from dlkit.datatypes.basic import FloatHyper, FloatRange, IntHyper, IntRange, StrHyper
from dlkit.utils.import_utils import load_class


class BaseSettings(BaseModel):
	model_config = ConfigDict(
		frozen=True,
		validate_default=True,
		extra='allow',
	)

	def to_dict_compatible_with(self, cls: type, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
		signature = inspect.signature(cls)

		return {
			field: value
			for field, value in self.model_dump().items()
			if field in signature.parameters.keys() and field not in exclude
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


class ClassSettings(BaseSettings):
	name: str
	module_path: str

	def construct_class_dynamic(self, settings_path: FilePath | None = None) -> type:
		class_name = load_class(self.name, self.module_path, settings_path)
		return class_name(**self.to_dict_compatible_with(class_name))
