from dynaconf import LazySettings
from pydantic import validate_call

from .general_settings import Settings


@validate_call(config={"arbitrary_types_allowed": True})
def dynaconf_to_settings(config: LazySettings) -> Settings:

    return Settings(
        **config.to_dict(),
    )
