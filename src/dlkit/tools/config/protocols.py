"""Settings protocols for interface segregation and dependency inversion."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from .components.model_components import ModelComponentSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathsSettings
from .session_settings import SessionSettings
from .training_settings import TrainingSettings as TrainingConfig


@runtime_checkable
class BaseSettingsProtocol(Protocol):
    """Protocol defining the minimal interface for all DLKit settings.

    This protocol follows ISP by exposing only core sections needed by all workflows.
    Specialized protocols extend this for workflow-specific requirements.

    Note: Core sections are expressed as read-only properties so that subclasses
    may narrow their declared types (e.g. ``SessionSettings`` instead of
    ``SessionSettings | None``) without violating protocol invariance.
    """

    # Core sections expressed as read-only properties for covariant compatibility
    @property
    @abstractmethod
    def SESSION(self) -> SessionSettings | None:
        """Session control settings."""
        ...

    @property
    @abstractmethod
    def MODEL(self) -> ModelComponentSettings | None:
        """Model component settings."""
        ...

    @property
    @abstractmethod
    def DATAMODULE(self) -> DataModuleSettings | None:
        """DataModule settings."""
        ...

    @property
    @abstractmethod
    def DATASET(self) -> DatasetSettings | None:
        """Dataset settings."""
        ...

    @property
    @abstractmethod
    def PATHS(self) -> PathsSettings | None:
        """Path override settings."""
        ...

    @property
    @abstractmethod
    def EXTRAS(self) -> object:
        """Free-form user settings."""
        ...

    @property
    @abstractmethod
    def is_training(self) -> bool:
        """True if running training (not inference)."""
        ...

    @property
    @abstractmethod
    def is_inference(self) -> bool:
        """True if running inference."""
        ...

    @property
    @abstractmethod
    def has_data_config(self) -> bool:
        """Check if dataflow configuration is available."""
        ...

    @abstractmethod
    def get_datamodule_config(self) -> DataModuleSettings:
        """Get datamodule configuration."""
        ...

    @abstractmethod
    def get_dataset_config(self) -> DatasetSettings:
        """Get dataset configuration."""
        ...


@runtime_checkable
class TrainingSettingsProtocol(BaseSettingsProtocol, Protocol):
    """Protocol for training-specific settings.

    Extends BaseSettingsProtocol with training-specific sections and capabilities.
    Follows ISP by only adding training-related methods.

    Note: TRAINING is required for training workflows, while MLFLOW and OPTUNA
    are optional plugins that can be enabled/disabled.
    """

    # Training-specific sections (TRAINING required, optimization plugins optional)
    @property
    @abstractmethod
    def TRAINING(self) -> TrainingConfig:
        """Core training configuration."""
        ...

    @property
    @abstractmethod
    def MLFLOW(self) -> MLflowSettings | None:
        """MLflow experiment tracking configuration."""
        ...

    @property
    @abstractmethod
    def OPTUNA(self) -> OptunaSettings | None:
        """Optuna hyperparameter optimization configuration."""
        ...

    @property
    @abstractmethod
    def mlflow_enabled(self) -> bool:
        """Check if MLflow is enabled and properly configured."""
        ...

    @property
    @abstractmethod
    def optuna_enabled(self) -> bool:
        """Check if Optuna is enabled and properly configured."""
        ...

    @property
    @abstractmethod
    def has_training_config(self) -> bool:
        """Check if training configuration is available."""
        ...

    @abstractmethod
    def get_training_config(self) -> TrainingConfig:
        """Get flattened training configuration."""
        ...


@runtime_checkable
class InferenceSettingsProtocol(BaseSettingsProtocol, Protocol):
    """Protocol for inference-specific settings.

    Extends BaseSettingsProtocol with inference-specific capabilities.
    Deliberately excludes training/optimization sections per ISP.
    """

    @property
    @abstractmethod
    def checkpoint_path(self) -> Path | str | None:
        """Get checkpoint path for inference."""
        ...


@runtime_checkable
class SettingsLoaderProtocol(Protocol):
    """Protocol for settings loading strategies.

    Enables dependency inversion by abstracting loading mechanism.
    """

    @abstractmethod
    def load_training_settings(self, config_path: Path | str) -> TrainingSettingsProtocol:
        """Load settings optimized for training workflows."""
        ...
