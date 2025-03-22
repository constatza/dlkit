from pydantic import validate_call
from loguru import logger
from dlkit.utils.system_utils import import_dynamically
from dlkit.settings import PrunerSettings


@validate_call
def initialize_pruner(settings: PrunerSettings):

    pruner_class = import_dynamically(settings.name, prepend="optuna.pruners")
    logger.info(f"Using pruner: {pruner_class.__name__}")
    return pruner_class(**settings.to_dict_compatible_with(pruner_class))
