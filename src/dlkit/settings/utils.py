from .classes import Settings
from dynaconf import LazySettings
from pydantic import validate_call


@validate_call(config={"arbitrary_types_allowed": True})
def dynaconf_to_settings(dynaconf_config: LazySettings) -> Settings:
    settings = Settings(**dynaconf_config.as_dict())
    return settings
