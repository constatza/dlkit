from pydantic import Field, model_validator

from .base_settings import BasicSettings
from .datamodule_settings import DataModuleSettings, DataloaderSettings
from .dataset import DatasetSettings
from .mlflow_settings import MLflowSettings
from .model_settings import ModelSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathSettings
from .trainer_settings import TrainerSettings
from .run_settings import RunSettings, RunMode


class Settings(BasicSettings):
    """Configuration settings for DLkit, encapsulating various components such as
    model, paths, MLflow, Optuna, trainer, and data settings.

    Attributes:
        RUN (RunSettings): Configuration for run settings.
        MODEL (ModelSettings): Configuration for model settings.
        PATHS (PathSettings): Configuration for paths settings.
        MLFLOW (MLflowSettings): Configuration for MLflow settings.
        OPTUNA (OptunaSettings): Configuration for Optuna settings.
        TRAINER (TrainerSettings): Configuration for trainer settings.
        DATAMODULE (DataModuleSettings): Configuration for datamodule settings.
        DATASET (DatasetSettings): Configuration for dataset settings.
        DATALOADER (DataloaderSettings): Configuration for dataloader settings.
    """

    MODEL: ModelSettings
    PATHS: PathSettings
    RUN: RunSettings = Field(default_factory=RunSettings, description="Run settings.")
    MLFLOW: MLflowSettings = Field(default_factory=MLflowSettings, description="MLflow settings.")
    OPTUNA: OptunaSettings = Field(default_factory=OptunaSettings, description="Optuna settings.")
    TRAINER: TrainerSettings = Field(
        default_factory=TrainerSettings, description="Lightning Trainer settings."
    )
    DATAMODULE: DataModuleSettings = Field(
        default_factory=DataModuleSettings, description="Lightning Datamodule settings."
    )
    DATASET: DatasetSettings = Field(
        default_factory=DatasetSettings, description="Dataset settings."
    )
    DATALOADER: DataloaderSettings = Field(
        default_factory=DataloaderSettings, description="Dataloader settings."
    )

    @model_validator(mode="after")
    def check_checkpoint_for_inference(self):
        """
        Ensure that when running in INFERENCE mode, checkpoint_path is provided.
        """
        if self.RUN.mode == RunMode.INFERENCE and not self.MODEL.checkpoint:
            raise ValueError(
                f"{self.PATHS.settings}: `checkpoint` path must be provided when running in inference mode."
            )
        return self
