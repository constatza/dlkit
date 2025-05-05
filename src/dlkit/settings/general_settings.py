from pydantic import Field, ValidationInfo, field_validator


from .base_settings import BaseSettings
from .paths_settings import PathSettings
from .trainer_settings import TrainerSettings
from .mlflow_settings import MLflowSettings
from .model_settings import ModelSettings
from .optuna_settings import OptunaSettings
from .data_settings import DataSettings


class Settings(BaseSettings):
    """
    Configuration settings for DLkit, encapsulating various components such as
    model, paths, MLflow, Optuna, trainer, and data settings.

    Attributes:
        MODEL (ModelSettings): Configuration for model settings.
        PATHS (PathSettings): Configuration for paths settings.
        MLFLOW (MLflowSettings): Configuration for MLflow settings.
        OPTUNA (OptunaSettings): Configuration for Optuna settings.
        TRAINER (TrainerSettings): Configuration for trainer settings.
        DATA (DataSettings): Configuration for data module settings.

    Methods:
        populate_is_autoencoder(cls, value, info): Automatically sets the
        'is_autoencoder' flag in DATA settings if the targets path is the same
        as the features path.
    """

    MODEL: ModelSettings
    PATHS: PathSettings
    MLFLOW: MLflowSettings
    OPTUNA: OptunaSettings = Field(
        default=OptunaSettings(), description="Optuna settings."
    )
    TRAINER: TrainerSettings = Field(
        default=TrainerSettings(), description="Trainer settings."
    )
    DATA: DataSettings = Field(..., description="Datamodule settings.")

    @field_validator("DATA")
    @classmethod
    def populate_targets_exist(cls, value: DataSettings, info: ValidationInfo):
        if info.data["PATHS"].targets == info.data["PATHS"].features:
            return value.model_copy(update={"targets_exist": True})
        return value
