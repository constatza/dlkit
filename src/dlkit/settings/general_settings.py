from pydantic import model_validator
from .base_settings import BaseSettings
from dlkit.settings import (
    ModelSettings,
    PathSettings,
    MLflowSettings,
    OptunaSettings,
    TrainerSettings,
    DatamoduleSettings,
)
from pydantic import field_validator, ValidationInfo
from pydantic import Field


class Settings(BaseSettings):
    """Settings for DLkit."""

    PATHS: PathSettings
    MODEL: ModelSettings
    MLFLOW: MLflowSettings = Field(default=MLflowSettings(), description="Model settings.")
    OPTUNA: OptunaSettings = Field(default=OptunaSettings(), description="Optuna settings.")
    TRAINER: TrainerSettings = Field(default=TrainerSettings(), description="Trainer settings.")
    DATAMODULE: DatamoduleSettings = Field(default=DatamoduleSettings(), description="Data module settings.")

    @field_validator("DATAMODULE", mode="after")
    @classmethod
    def populate_is_autoencoder(cls, value: DatamoduleSettings, info: ValidationInfo):
        if info.data["PATHS"].targets is None:
            return value.model_copy(update={"is_autoencoder": True})
        return value
