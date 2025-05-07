from loguru import logger
from pydantic import validate_call

from dlkit.settings import PrunerSettings
from dlkit.utils.system_utils import import_dynamic


@validate_call
def initialize_pruner(settings: PrunerSettings):
	pruner_class = import_dynamic(settings.name, prepend='optuna.pruners')
	logger.info(f'Using pruner: {pruner_class.__name__}')
	return pruner_class(**settings.to_dict_compatible_with(pruner_class))
