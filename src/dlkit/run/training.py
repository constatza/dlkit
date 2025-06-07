from .vanilla_training import train_vanilla
from .mlflow_training import train_mlflow
from .optuna_training import train_optuna
from loguru import logger
from dlkit.io.settings import load_validated_settings
from dlkit.datatypes.learning import ModelState


def train(settings, optuna: bool = True, mlflow: bool = True):
    if settings.OPTUNA.enable and optuna:
        logger.info("Using Optuna for hyperparameter optimization.")
        return train_optuna(settings)
    if settings.MLFLOW.enable and mlflow:
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
