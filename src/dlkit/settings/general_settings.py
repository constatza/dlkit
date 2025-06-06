from pydantic import Field

from .base_settings import BaseSettings
from .datamodule_settings import DataModuleSettings, DatasetSettings, DataloaderSettings
from .mlflow_settings import MLflowSettings
from .model_settings import ModelSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathSettings
from .trainer_settings import TrainerSettings


class Settings(BaseSettings):
    """Configuration settings for DLkit, encapsulating various components such as
    model, paths, MLflow, Optuna, trainer, and data settings.

    Attributes:
        MODEL (ModelSettings): Configuration for model settings.
        PATHS (PathSettings): Configuration for paths settings.
        MLFLOW (MLflowSettings): Configuration for MLflow settings.
        OPTUNA (OptunaSettings): Configuration for Optuna settings.
        TRAINER (TrainerSettings): Configuration for trainer settings.
        DATAMODULE (DataModuleSettings): Configuration for datamodule settings.
        DATASET (DatasetSettings): Configuration for dataset settings.
        DATALOADER (DataloaderSettings): Configuration for dataloader settings.
        seed (int): Random seed for reproducibility.
    """

    MODEL: ModelSettings
    PATHS: PathSettings
    MLFLOW: MLflowSettings
    OPTUNA: OptunaSettings = Field(default=OptunaSettings(), description="Optuna settings.")
    TRAINER: TrainerSettings = Field(default=TrainerSettings(), description="Trainer settings.")
    DATAMODULE: DataModuleSettings = Field(..., description="Datamodule settings.")
    DATASET: DatasetSettings = Field(..., description="Dataset settings.")
    DATALOADER: DataloaderSettings = Field(DataloaderSettings(), description="Dataloader settings.")
    seed: int = Field(default=1, description="Random seed for reproducibility.")
    precision: str = Field(default="medium", description="Precision for float32 matmul operations.")
