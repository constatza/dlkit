from dynaconf import LazySettings
from pydantic import validate_call

from .general_settings import Settings


@validate_call(config={"arbitrary_types_allowed": True})
def dynaconf_to_settings(config: LazySettings) -> Settings:
    """Convert a LazySettings object to a Settings object."""
    return Settings(**config.to_dict())
