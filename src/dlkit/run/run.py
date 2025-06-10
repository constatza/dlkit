from typing import Literal
from loguru import logger
from dlkit.io.settings import load_validated_settings
from dlkit.datatypes.run import ModelState
from dlkit.settings import RunMode, RunSettings
from dlkit.setup.model_state import build_model_state
from dlkit.settings import Settings
from .vanilla_training import train_simple
from .mlflow_training import train_mlflow
from .optuna_training import train_optuna


def run(
    settings: Settings,
    mode: Literal["training", "inference", "mlflow", "optuna"] | None = None,
):
    if mode:
        settings = update_mode(settings, mode)
    if settings.RUN.mode is RunMode.INFERENCE:
        logger.info("Running inference.")
        return build_model_state(settings)

    if settings.RUN.mode is RunMode.MLFLOW:
        logger.info("Using MLflow for logging.")
        return train_mlflow(settings)

    if settings.RUN.mode is RunMode.OPTUNA:
        logger.info("Using Optuna for hyperparameter optimization.")
        return train_optuna(settings)

    return train_simple(settings)


def run_from_path(
    settings_path: str, mode: Literal["training", "inference", "mlflow", "optuna"] | None = None
) -> ModelState:
    """Loads the settings from a file and runs the training or inference process.

    Args:
        settings_path (str): The path to the settings file.
        mode (Literal["training", "inference", "mlflow", "optuna"] | None, optional): If given, the mode overrides
            the mode in the settings file.

    Returns:
        ModelState: The state of the model after training or inference.
    """
    settings = load_validated_settings(settings_path)
    return run(settings, mode=mode)


def update_mode(
    settings: Settings, mode: Literal["training", "inference", "mlflow", "optuna"] | None
) -> Settings:
    return settings.model_copy(
        update={"RUN": RunSettings(mode=RunMode(mode), **settings.RUN.model_dump(exclude={"mode"}))}
    )
