from .vanilla_training import train_vanilla
from .mlflow_training import train_mlflow
from .optuna_training import train_optuna
from loguru import logger
from dlkit.io.settings import load_validated_settings
from dlkit.datatypes.learning import TrainingState
from dlkit.settings import Settings


def train(settings, optuna: bool = False, mlflow: bool = False):
    if settings.OPTUNA.enable or optuna:
        logger.info("Using Optuna for hyperparameter optimization.")
        return train_optuna(settings)
    if settings.MLFLOW.enable or mlflow:
        logger.info("Using MLflow for logging.")
        return train_mlflow(settings)
    return train_vanilla(settings)


def train_state_from_path(settings_path: str) -> tuple[TrainingState, Settings]:
    settings = load_validated_settings(settings_path)
    return train_vanilla(settings), settings
