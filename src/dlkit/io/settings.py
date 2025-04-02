from dynaconf import Dynaconf, LazySettings
from pydantic import validate_call, FilePath
from dlkit.settings.general_settings import Settings
from dlkit.settings.utils import dynaconf_to_settings


@validate_call
def load_validated_settings(file_path: FilePath) -> Settings:
    config = Dynaconf(settings_files=[file_path], envvar_prefix="DLKIT")
    dlkit_settings = dynaconf_to_settings(config)
    return dlkit_settings
