from dynaconf import LazySettings
from pydantic import validate_call, ConfigDict

from .general_settings import Settings


@validate_call(config=ConfigDict(arbitrary_types_allowed=True))
def dynaconf_to_settings(config: LazySettings) -> Settings:
    """Convert a LazySettings object to a Settings object."""
    try:
        if config.get("optuna.model") and config.get("optuna.enable"):
            optuna_model = config.get("optuna.model")
            config.update({f"model.{key}": value for key, value in optuna_model.items()})
        as_dict = config.to_dict()
    except ValueError as e:
        raise ValueError(f"Configuration file is not valid - {e}")
    return Settings(**as_dict)
