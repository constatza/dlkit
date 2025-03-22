from pydantic import model_validator
from .base_settings import BaseSettings
from dlkit.settings import (
    ModelSettings,
    Paths,
    MLflowSettings,
    OptunaSettings,
    TrainerSettings,
    DatamoduleSettings,
)


class Settings(BaseSettings):
    """Settings for DLkit."""

    MODEL: ModelSettings
    PATHS: Paths
    MLFLOW: MLflowSettings
    OPTUNA: OptunaSettings
    TRAINER: TrainerSettings
    DATAMODULE: DatamoduleSettings

    @model_validator(mode="before")
    def is_autoencoder(cls, data: dict) -> dict:
        if not "PATHS" in data and not "MODEL" in data and not "DATAMODULE" in data:
            raise ValueError("PATHS, MODEL, and DATAMODULE must be defined.")

        if "targets" not in data["PATHS"] and "features" in data["PATHS"]:
            data["MODEL"].update({"is_autoencoder": True})
            data["DATAMODULE"].update({"autoencoder_dataset": True})

        return data
