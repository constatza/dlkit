"""Flattened general settings with SOLID principles and top-level configuration areas."""

from __future__ import annotations

from typing import Self, cast

from pydantic import Field, FilePath, model_validator, validate_call

from .components.model_components import ModelComponentSettings
from .core.base_settings import BasicSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .extras_settings import ExtrasSettings
from .generative_settings import GenerativeSettings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathsSettings
from .session_settings import SessionSettings
from .training_settings import TrainingSettings as TrainingConfig


class GeneralSettings(BasicSettings):
    """Flattened configuration settings for DLKit with SOLID principles.

    This implements a flattened architecture that eliminates deep nesting
    and follows SOLID principles. Main functional settings are at top-level
    with only library-specific configurations remaining nested.

    Top-level fields use CAPITALS for dynaconf compatibility.

    Flattened Architecture:
    - MLFLOW: MLflow experiment tracking
    - OPTUNA: Hyperparameter optimization
    - DATAMODULE: Data loading configuration
    - DATASET: Dataset configuration
    - TRAINING: Core training settings
    - PATHS: Standardized paths with automatic resolution (optional)
    - EXTRAS: Free-form user-defined helper settings (ignored by core)
    - GENERATIVE: Generative algorithm configuration (optional)

    Preserved Nesting (Library-Specific Only):
    - TRAINING.trainer: PyTorch Lightning trainer settings
    - TRAINING.optimizer: Optimizer and scheduler settings

    Args:
        SESSION: Session mode control (simplified)
        MLFLOW: MLflow experiment tracking configuration
        OPTUNA: Hyperparameter optimization configuration
        DATAMODULE: Data loading and processing configuration
        DATASET: Dataset-specific configuration
        TRAINING: Core training configuration with nested library settings
        PATHS: Optional standardized paths with automatic resolution
        EXTRAS: Arbitrary user-defined values for custom scripts/tools
        GENERATIVE: Optional generative algorithm configuration
    """

    # Core infrastructure settings
    SESSION: SessionSettings = Field(
        default_factory=SessionSettings, description="Session mode control and execution settings"
    )

    # Prefer shallow hierarchy: expose MODEL at the top level
    MODEL: ModelComponentSettings | None = Field(
        default=None, description="Model configuration (preferred at top-level)"
    )

    # Flattened functional settings
    MLFLOW: MLflowSettings | None = Field(
        default=None,
        description="MLflow experiment tracking configuration. "
        "The presence of this section enables MLflow tracking.",
    )
    OPTUNA: OptunaSettings | None = Field(
        default_factory=OptunaSettings,
        description="Optuna hyperparameter optimization configuration",
    )
    DATAMODULE: DataModuleSettings | None = Field(
        default_factory=DataModuleSettings,
        description="Data loading and processing configuration",
    )
    DATASET: DatasetSettings | None = Field(
        default_factory=DatasetSettings, description="Dataset-specific configuration"
    )
    TRAINING: TrainingConfig | None = Field(
        default_factory=TrainingConfig,
        description="Core training configuration with nested library settings",
    )
    # Optional standardized paths with automatic resolution
    PATHS: PathsSettings | None = Field(
        default=None,
        description="Optional standardized paths with automatic resolution relative to root_dir",
    )
    # Optional, free-form user extras (ignored by core libraries)
    EXTRAS: ExtrasSettings | None = Field(
        default=None,
        description="Optional free-form helper options for user scripts; ignored by core",
    )
    # Optional generative algorithm configuration
    GENERATIVE: GenerativeSettings | None = Field(
        default=None,
        description="Optional generative algorithm configuration. "
        "When present, selects a generative build strategy. "
        "Supported algorithms: 'flow_matching', 'cnf'.",
    )

    @model_validator(mode="after")
    def validate_inference_checkpoint(self):
        """Ensure inference mode has checkpoint path configured using top-level MODEL only."""
        if self.SESSION.is_inference_mode:
            if not (self.MODEL and self.MODEL.checkpoint):
                raise ValueError(
                    "Checkpoint path must be provided when running in inference mode. "
                    "Add 'checkpoint = \"/path/to/model.ckpt\"' under [MODEL] section."
                )
        return self

    @classmethod
    @validate_call
    def from_toml_file(cls, filepath: FilePath | str) -> Self:
        """Factory: load and validate settings from a TOML config file using clean architecture.

        This method uses the new TOML config loading system with SOLID principles
        and functional programming patterns. Recommended for new code.

        Args:
            filepath: Path to TOML configuration file

        Returns:
            Self: Parsed and validated settings object

        Raises:
            ValueError: If the config cannot be loaded or validated
        """
        from dlkit.tools.io.config import load_config

        try:
            config = load_config(filepath, cls)
            if isinstance(config, cls):
                return cast(Self, config)
            # Handle dict case
            return cls.model_validate(config)

        except Exception as e:
            raise ValueError(f"Failed to load configuration from {filepath}: {e}") from e

    @property
    def is_training(self) -> bool:
        """True if running training (not inference)."""
        return not self.SESSION.inference

    @property
    def is_inference(self) -> bool:
        """True if running inference."""
        return bool(self.SESSION.inference)

    @property
    def is_testing(self) -> bool:
        """Testing mode is not used in the simplified model."""
        return False

    # Convenience properties for flattened access
    @property
    def mlflow_enabled(self) -> bool:
        """Check if MLflow is enabled and properly configured.

        Returns:
            bool: True if MLflow tracking is active
        """
        return self.MLFLOW is not None

    @property
    def optuna_enabled(self) -> bool:
        """Check if Optuna is enabled and properly configured.

        Returns:
            bool: True if Optuna optimization is active
        """
        return self.OPTUNA is not None and self.OPTUNA.enabled

    @property
    def has_training_config(self) -> bool:
        """Check if training configuration is available.

        Returns:
            bool: True if training settings are configured
        """
        return self.TRAINING is not None

    @property
    def has_data_config(self) -> bool:
        """Check if dataflow configuration is available.

        Returns:
            bool: True if datamodule and dataset are configured
        """
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_training_config(self) -> TrainingConfig:
        """Get flattened training configuration.

        Returns:
            TrainingConfig: Training settings

        Raises:
            ValueError: If no training configuration available
        """
        if not self.TRAINING:
            raise ValueError("No training configuration available")
        return self.TRAINING

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
