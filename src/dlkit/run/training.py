from .vanilla_training import train_vanilla
from .mlflow_training import train_mlflow
from .optuna_training import train_optuna
from loguru import logger
from dlkit.io.settings import load_validated_settings
from dlkit.datatypes.run import ModelState
from dlkit.settings import RunMode


def train(settings, optuna: bool = False, mlflow: bool = False):
    if settings.RUN.mode is RunMode.OPTUNA or optuna:
        logger.info("Using Optuna for hyperparameter optimization.")
        return train_optuna(settings)
    if settings.RUN.mode is RunMode.MLFLOW or mlflow:
        logger.info("Using MLflow for logging.")
        return train_mlflow(settings)
    return train_vanilla(settings)


def train_state_from_path(
    settings_path: str, optuna: bool = True, mlflow: bool = True, inference: bool = False
) -> ModelState:
    settings = load_validated_settings(settings_path)
    return train(
        settings,
        optuna=optuna,
        mlflow=mlflow,
    )
