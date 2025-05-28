from .vanilla_training import train_vanilla
from .mlflow_training import train_mlflow
from .optuna_training import train_optuna
from loguru import logger


def train(settings, optuna: bool = False, mlflow: bool = False):
    if settings.OPTUNA.enable or optuna:
        logger.info("Using Optuna for hyperparameter optimization.")
        return train_optuna(settings)
    if settings.MLFLOW.enable or mlflow:
        logger.info("Using MLflow for logging.")
        return train_mlflow(settings)
    return train_vanilla(settings)
