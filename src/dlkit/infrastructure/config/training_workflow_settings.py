"""Training workflow settings."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .training_settings import TrainingSettings
from .workflow_settings_base import BaseWorkflowSettings


class TrainingWorkflowSettings(BaseWorkflowSettings):
    """Settings class specialized for training workflows."""

    model_config = SettingsConfigDict(env_prefix="DLKIT_", env_nested_delimiter="__")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type,
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return (init_settings, env_settings)

    TRAINING: TrainingSettings | None = Field(
        default=None,
        description="Core training configuration with nested library settings",
    )
    MLFLOW: MLflowSettings | None = Field(
        default=None,
        description="MLflow experiment tracking configuration",
    )
    OPTUNA: OptunaSettings | None = Field(
        default=None,
        description="Optuna hyperparameter optimization configuration",
    )

    _workflow_type: ClassVar[str] = "training"

    @property
    def mlflow_enabled(self) -> bool:
        return self.MLFLOW is not None

    @property
    def optuna_enabled(self) -> bool:
        return self.OPTUNA is not None and self.OPTUNA.enabled

    @property
    def has_training_config(self) -> bool:
        return self.TRAINING is not None

    def get_training_config(self) -> TrainingSettings:
        if not self.TRAINING:
            raise ValueError("No training configuration available")
        return self.TRAINING
