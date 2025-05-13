from pydantic import Field

from .base_settings import BaseSettings
from .data_settings import DataSettings
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
	    DATA (DataSettings): Configuration for data module settings.
	    seed (int): Random seed for reproducibility.
	"""

	MODEL: ModelSettings
	PATHS: PathSettings
	MLFLOW: MLflowSettings
	OPTUNA: OptunaSettings = Field(default=OptunaSettings(), description='Optuna settings.')
	TRAINER: TrainerSettings = Field(default=TrainerSettings(), description='Trainer settings.')
	DATA: DataSettings = Field(..., description='Datamodule settings.')
	seed: int = Field(default=42, description='Random seed for reproducibility.')
