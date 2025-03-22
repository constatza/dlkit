from collections.abc import Sequence

from dynaconf import LazySettings
from pydantic import validate_call
from optuna import Trial
from optuna.distributions import CategoricalChoiceType

from .types import IntRange, FloatRange, IntHyper, FloatHyper, StrHyper
from .general_settings import Settings
from dlkit.settings import (
    ModelSettings,
    Paths,
    MLflowSettings,
    OptunaSettings,
    TrainerSettings,
    DatamoduleSettings,
)


@validate_call(config={"arbitrary_types_allowed": True})
def dynaconf_to_settings(dynaconf_config: LazySettings) -> Settings:
    return Settings(**dynaconf_config.to_dict())
