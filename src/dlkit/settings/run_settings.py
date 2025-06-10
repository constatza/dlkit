from enum import StrEnum
from .base_settings import BaseSettings
from pydantic import Field


class RunMode(StrEnum):
    TRAINING = "training"
    TESTING = "testing"
    INFERENCE = "inference"
    MLFLOW = "mlflow"
    OPTUNA = "optuna"


class RunSettings(BaseSettings):
    name: str = Field("run", description="Name of the run.")
    mode: RunMode = Field(RunMode.TRAINING, description="Mode of the run.")
    continue_training: bool = Field(
        default=False, description="Whether to continue training from a checkpoint."
    )
    seed: int = Field(default=1, description="Random seed for reproducibility.")
    precision: str = Field(default="medium", description="Precision for float32 matmul operations.")
