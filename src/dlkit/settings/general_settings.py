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
from pydantic import field_validator, ValidationInfo


class Settings(BaseSettings):
    """Settings for DLkit."""

    PATHS: Paths
    MODEL: ModelSettings
    MLFLOW: MLflowSettings
    OPTUNA: OptunaSettings
    TRAINER: TrainerSettings
    DATAMODULE: DatamoduleSettings

    @field_validator("DATAMODULE", mode="after", check_fields=True)
    @classmethod
    def populate_is_autoencoder(cls, value: DatamoduleSettings, info: ValidationInfo):
        if info.data["PATHS"].targets is None:
            return value.model_copy(update={"is_autoencoder": True})
        return value
