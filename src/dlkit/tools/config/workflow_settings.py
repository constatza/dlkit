"""Workflow-specific settings classes following SOLID principles."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, model_validator

from .core.base_settings import BasicSettings
from .session_settings import SessionSettings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .training_settings import TrainingSettings as TrainingConfig
from .components.model_components import ModelComponentSettings
from .extras_settings import ExtrasSettings
from .paths_settings import PathsSettings


class BaseWorkflowSettings(BasicSettings):
    """Base settings class implementing common functionality for all workflows.

    This class follows SRP by handling only core sections common to all workflows.
    It's open for extension (OCP) through inheritance and closed for modification.
    All sections are optional to support flexible partial loading.
    """

    # Core infrastructure settings (can be optional for partial loading)
    SESSION: SessionSettings | None = Field(
        default=None, description="Session mode control and execution settings"
    )

    # Core functional settings (common to all workflows)
    MODEL: ModelComponentSettings | None = Field(
        default=None, description="Model configuration"
    )
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading and processing configuration",
    )
    DATASET: DatasetSettings | None = Field(
        default=None, description="Dataset-specific configuration"
    )

    # Optional common settings
    PATHS: PathsSettings | None = Field(
        default=None,
        description="Optional standardized paths with automatic resolution relative to root_dir",
    )
    EXTRAS: ExtrasSettings | None = Field(
        default=None,
        description="Optional free-form helper options for user scripts; ignored by core",
    )

    # Workflow identification (template method pattern support)
    _workflow_type: ClassVar[str] = "base"

    @property
    def is_training(self) -> bool:
        """True if running training (not inference)."""
        return not (self.SESSION and self.SESSION.inference)

    @property
    def is_inference(self) -> bool:
        """True if running inference."""
        return bool(self.SESSION and self.SESSION.inference)

    @property
    def is_testing(self) -> bool:
        """Testing mode is not used in the simplified model."""
        return False

    @property
    def has_data_config(self) -> bool:
        """Check if dataflow configuration is available."""
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_datamodule_config(self) -> DataModuleSettings:
        """Get datamodule configuration.

        Returns:
            DataModuleSettings: DataModule settings

        Raises:
            ValueError: If no datamodule configuration available
        """
        if not self.DATAMODULE:
            raise ValueError("No datamodule configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        """Get dataset configuration.

        Returns:
            DatasetSettings: Dataset settings

        Raises:
            ValueError: If no dataset configuration available
        """
        if not self.DATASET:
            raise ValueError("No dataset configuration available")
        return self.DATASET


class TrainingWorkflowSettings(BaseWorkflowSettings):
    """Settings class specialized for training workflows.

    Follows SRP by handling only training-specific configuration.
    Implements TrainingSettingsProtocol for type safety and ISP compliance.
    Uses default_factory for typical training workflows but allows None for partial loading.
    """

    # Core infrastructure settings (required for training)
    # Note: These override the base class but are assignment-compatible
    # Base allows None, we require specific instances - this is LSP compatible
    def __init__(self, **data):
        # Ensure required fields have defaults if not provided
        if 'SESSION' not in data or data['SESSION'] is None:
            data['SESSION'] = SessionSettings()
        if 'DATAMODULE' not in data or data['DATAMODULE'] is None:
            data['DATAMODULE'] = DataModuleSettings()
        if 'DATASET' not in data or data['DATASET'] is None:
            data['DATASET'] = DatasetSettings()
        if 'TRAINING' not in data or data['TRAINING'] is None:
            data['TRAINING'] = TrainingConfig()
        super().__init__(**data)

    # Training-specific sections (required)
    TRAINING: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Core training configuration with nested library settings",
    )

    # Optional training-specific sections
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
        """Check if MLflow is enabled and properly configured."""
        return self.MLFLOW is not None and self.MLFLOW.enabled

    @property
    def optuna_enabled(self) -> bool:
        """Check if Optuna is enabled and properly configured."""
        return self.OPTUNA is not None and self.OPTUNA.enabled

    @property
    def has_training_config(self) -> bool:
        """Check if training configuration is available."""
        return self.TRAINING is not None

    def get_training_config(self) -> TrainingConfig:
        """Get flattened training configuration.

        Returns:
            TrainingSettings: Training settings

        Raises:
            ValueError: If no training configuration available
        """
        if not self.TRAINING:
            raise ValueError("No training configuration available")
        return self.TRAINING


class InferenceWorkflowSettings(BaseWorkflowSettings):
    """Settings class specialized for inference workflows.

    Follows SRP by handling only inference-specific configuration.
    Deliberately excludes training/optimization sections per ISP.
    Implements InferenceSettingsProtocol for type safety.
    """

    # Core infrastructure settings (required for inference)
    # Note: These override the base class but are assignment-compatible
    def __init__(self, **data):
        # Ensure required fields have defaults if not provided
        if 'SESSION' not in data or data['SESSION'] is None:
            data['SESSION'] = SessionSettings()
        if 'DATAMODULE' not in data or data['DATAMODULE'] is None:
            data['DATAMODULE'] = DataModuleSettings()
        if 'DATASET' not in data or data['DATASET'] is None:
            data['DATASET'] = DatasetSettings()
        super().__init__(**data)

    _workflow_type: ClassVar[str] = "inference"

    @model_validator(mode="after")
    def validate_inference_checkpoint(self):
        """Ensure inference mode has checkpoint path configured."""
        if self.SESSION and self.SESSION.inference:
            if not (self.MODEL and self.MODEL.checkpoint):
                raise ValueError(
                    "Checkpoint path must be provided when running in inference mode. "
                    "Add 'checkpoint = \"/path/to/model.ckpt\"' under [MODEL] section."
                )
        return self

    @property
    def checkpoint_path(self) -> Path | str | None:
        """Get checkpoint path for inference."""
        return self.MODEL.checkpoint if self.MODEL else None


# Expose type aliases for clean imports
BaseSettings = BaseWorkflowSettings
TrainingSettings = TrainingWorkflowSettings
InferenceSettings = InferenceWorkflowSettings