"""Workflow-specific config models with eager validation and optional sections.

This module replaces the lazy loading pattern with eager Pydantic validation while
maintaining section-level programmatic override flexibility. Config sections are
explicitly marked as Optional[T] to support partial TOML configs with programmatic
injection.

Design Pattern: "Partial Models" with Completeness Validation
- Load time: Eager validation of present fields (fail-fast on typos/types)
- Pre-build: Completeness validation ensures all required sections injected
- Runtime: Type system guarantees section presence after completeness check

Architecture Principles:
- SOLID compliant: Workflow-specific configs (SRP), extensible via inheritance (OCP)
- Type-safe: Optional[T] makes programmatic injection explicit
- Fail-fast: Pydantic validates immediately, completeness validators before build
- Immutable: Public updates use ``patch()`` / ``update_with()`` to produce new instances
"""

from __future__ import annotations

import warnings
from typing import Any

from pydantic import Field, model_validator

from .core.base_settings import BasicSettings
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .mlflow_settings import MLflowSettings
from .model_components import ModelComponentSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathsSettings
from .session_settings import SessionSettings
from .training_settings import TrainingSettings as TrainingConfig


class TrainingWorkflowConfig(BasicSettings):
    """Configuration for training workflows with eager validation.

    Required Sections (at load time):
        - SESSION: Session control (name, seed, root_dir, precision)
        - TRAINING: Training config (epochs, optimizer, loss, metrics)

    Optional Sections (can be injected programmatically):
        - DATAMODULE: Data loading config (can be provided via API)
        - DATASET: Dataset config (can be provided via API)
        - MODEL: Model config (can be provided via API)
        - MLFLOW: Experiment tracking (defaults to disabled)
        - OPTUNA: Hyperparameter optimization (defaults to disabled)
        - PATHS: Custom path overrides (falls back to SESSION.root_dir)
        - EXTRAS: Free-form user settings (ignored by core)

    Validation Strategy:
        1. Load time: Pydantic validates all present fields eagerly
        2. Pre-build: Completeness validator ensures DATAMODULE/DATASET/MODEL present
        3. Build time: BuildFactory constructs components

    Example Usage:
        ```python
        # Load partial config from TOML (eager validation of present fields)
        config = TrainingWorkflowConfig.model_validate(toml_dict)

        # Inject sections programmatically (also validated eagerly)
        config = config.patch({"DATASET": DatasetSettings(features=(...), targets=(...))})

        # Validate completeness before building
        validate_training_config_complete(config)  # Raises if sections missing

        # Build components
        components = BuildFactory().build_components(config)
        ```
    """

    # Required sections (must be in TOML)
    SESSION: SessionSettings = Field(
        ..., description="Session mode control and execution settings (required)"
    )
    TRAINING: TrainingConfig = Field(..., description="Core training configuration (required)")

    # Optional sections (can be injected programmatically)
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading config (optional at load, required at build)",
    )
    DATASET: DatasetSettings | None = Field(
        default=None,
        description="Dataset config (optional at load, required at build)",
    )
    MODEL: ModelComponentSettings | None = Field(
        default=None,
        description="Model config (optional at load, required at build)",
    )

    # Optional tracking/optimization
    MLFLOW: MLflowSettings | None = Field(
        default=None,
        description="MLflow experiment tracking (presence of section enables tracking)",
    )
    OPTUNA: OptunaSettings = Field(
        default_factory=lambda: OptunaSettings(enabled=False),
        description="Optuna hyperparameter optimization (defaults to disabled)",
    )

    # Optional user sections
    PATHS: PathsSettings | None = Field(
        default=None,
        description="Custom path overrides (falls back to SESSION.root_dir)",
    )
    EXTRAS: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form user settings (ignored by core)",
    )

    # Convenience properties
    @property
    def mlflow_enabled(self) -> bool:
        """Check if MLflow tracking is enabled."""
        return self.MLFLOW is not None

    @property
    def optuna_enabled(self) -> bool:
        """Check if Optuna optimization is enabled."""
        return self.OPTUNA.enabled

    @property
    def has_complete_data_config(self) -> bool:
        """Check if both DATAMODULE and DATASET are present."""
        return self.DATAMODULE is not None and self.DATASET is not None

    @property
    def has_model_config(self) -> bool:
        """Check if MODEL section is present."""
        return self.MODEL is not None

    @property
    def is_training(self) -> bool:
        """True when running in training mode."""
        return self.SESSION.workflow == "train" if self.SESSION else True

    @property
    def is_inference(self) -> bool:
        """True when running in inference mode."""
        return self.SESSION.workflow == "inference" if self.SESSION else False

    @property
    def has_data_config(self) -> bool:
        """True when both DATAMODULE and DATASET sections are present."""
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_datamodule_config(self) -> DataModuleSettings:
        """Return the DATAMODULE configuration section.

        Returns:
            DataModuleSettings instance.

        Raises:
            ValueError: If DATAMODULE section is absent.
        """
        if self.DATAMODULE is None:
            raise ValueError("No DATAMODULE configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        """Return the DATASET configuration section.

        Returns:
            DatasetSettings instance.

        Raises:
            ValueError: If DATASET section is absent.
        """
        if self.DATASET is None:
            raise ValueError("No DATASET configuration available")
        return self.DATASET


class InferenceWorkflowConfig(BasicSettings):
    """Configuration for inference workflows with eager validation.

    Required Sections (at load time):
        - SESSION: Session control with workflow="inference"
        - MODEL: Model config with checkpoint path

    Optional Sections (for batch inference):
        - DATAMODULE: Data loading config (for batch inference)
        - DATASET: Dataset config (for batch inference)
        - PATHS: Custom path overrides
        - EXTRAS: Free-form user settings

    For batch prediction, provide both DATASET and DATAMODULE. For
    single-sample inference, omit both and use ``load_model()`` directly.

    For batch prediction, provide both DATASET and DATAMODULE sections.
    For single-sample checkpoint inference, omit both and use ``load_model()``
    or ``load_model_from_settings()`` directly.

    Note: Training-specific sections (TRAINING, MLFLOW, OPTUNA) are deliberately
    excluded per Interface Segregation Principle.

    Example Usage:
        ```python
        # Load config
        config = InferenceWorkflowConfig.model_validate(toml_dict)

        # Validate checkpoint exists
        if config.MODEL.checkpoint is None:
            raise ValueError("Checkpoint required for inference")

        # Use with predictor API
        predictor = load_model(config.MODEL.checkpoint)
        ```
    """

    # Required sections
    SESSION: SessionSettings = Field(
        ..., description="Session control with workflow='inference' (required)"
    )
    MODEL: ModelComponentSettings = Field(
        ..., description="Model config with checkpoint (required)"
    )

    # Optional sections (for batch inference)
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading config (optional, for batch inference)",
    )
    DATASET: DatasetSettings | None = Field(
        default=None,
        description="Dataset config (optional, for batch inference)",
    )

    # Optional user sections
    PATHS: PathsSettings | None = Field(
        default=None,
        description="Custom path overrides",
    )
    EXTRAS: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form user settings (ignored by core)",
    )

    @property
    def has_dataset_config(self) -> bool:
        """Check if dataset-backed batch prediction config is available."""
        return self.DATAMODULE is not None and self.DATASET is not None

    @property
    def is_training(self) -> bool:
        """Always False for inference workflows."""
        return False

    @property
    def is_inference(self) -> bool:
        """Always True for inference workflows."""
        return True

    @property
    def has_data_config(self) -> bool:
        """True when both DATAMODULE and DATASET sections are present."""
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_datamodule_config(self) -> DataModuleSettings:
        """Return the DATAMODULE configuration section.

        Returns:
            DataModuleSettings instance.

        Raises:
            ValueError: If DATAMODULE section is absent.
        """
        if self.DATAMODULE is None:
            raise ValueError("No DATAMODULE configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        """Return the DATASET configuration section.

        Returns:
            DatasetSettings instance.

        Raises:
            ValueError: If DATASET section is absent.
        """
        if self.DATASET is None:
            raise ValueError("No DATASET configuration available")
        return self.DATASET


class OptimizationWorkflowConfig(BasicSettings):
    """Configuration for hyperparameter optimization workflows with eager validation.

    Required Sections (at load time):
        - SESSION: Session control with workflow="optimize"
        - TRAINING: Training config
        - OPTUNA: Optimization config with enabled=true

    Optional Sections (can be injected programmatically):
        - DATAMODULE: Data loading config
        - DATASET: Dataset config
        - MODEL: Model config
        - MLFLOW: Experiment tracking (can track optimization studies)
        - PATHS: Custom path overrides
        - EXTRAS: Free-form user settings

    Example Usage:
        ```python
        # Load config
        config = OptimizationWorkflowConfig.model_validate(toml_dict)

        # Validate Optuna enabled
        if not config.OPTUNA.enabled:
            raise ValueError("OPTUNA.enabled must be true")

        # Run optimization
        study = optimization_service.execute_optimization(config)
        ```
    """

    # Required sections
    SESSION: SessionSettings = Field(..., description="Session mode control (required)")
    TRAINING: TrainingConfig = Field(..., description="Core training configuration (required)")
    OPTUNA: OptunaSettings = Field(
        ..., description="Optuna optimization config with enabled=true (required)"
    )

    # Optional sections (can be injected programmatically)
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading config (optional at load, required at build)",
    )
    DATASET: DatasetSettings | None = Field(
        default=None,
        description="Dataset config (optional at load, required at build)",
    )
    MODEL: ModelComponentSettings | None = Field(
        default=None,
        description="Model config (optional at load, required at build)",
    )

    # Optional tracking
    MLFLOW: MLflowSettings | None = Field(
        default=None,
        description="MLflow experiment tracking (presence of section enables tracking)",
    )

    # Optional user sections
    PATHS: PathsSettings | None = Field(
        default=None,
        description="Custom path overrides",
    )
    EXTRAS: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form user settings (ignored by core)",
    )

    @model_validator(mode="after")
    def _validate_optuna_model_keys(self) -> OptimizationWorkflowConfig:
        """Warn when OPTUNA.model contains keys outside the MODEL schema."""
        if self.MODEL is None or not self.OPTUNA.model:
            return self

        model_fields = set(type(self.MODEL).model_fields.keys())
        model_fields.update((self.MODEL.model_extra or {}).keys())
        optuna_keys = set(self.OPTUNA.model.keys())
        unknown = optuna_keys - model_fields - {"lr", "weight_decay"}
        if unknown:
            warnings.warn(
                f"OPTUNA.model contains keys not in MODEL: {sorted(unknown)}. "
                "These ranges will be ignored.",
                UserWarning,
                stacklevel=2,
            )
        return self

    @property
    def mlflow_enabled(self) -> bool:
        """Check if MLflow tracking is enabled."""
        return self.MLFLOW is not None

    @property
    def has_complete_data_config(self) -> bool:
        """Check if both DATAMODULE and DATASET are present."""
        return self.DATAMODULE is not None and self.DATASET is not None

    @property
    def has_model_config(self) -> bool:
        """Check if MODEL section is present."""
        return self.MODEL is not None

    @property
    def is_training(self) -> bool:
        """True when running in training mode."""
        return self.SESSION.workflow == "train" if self.SESSION else True

    @property
    def is_inference(self) -> bool:
        """True when running in inference mode."""
        return self.SESSION.workflow == "inference" if self.SESSION else False

    @property
    def has_data_config(self) -> bool:
        """True when both DATAMODULE and DATASET sections are present."""
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_datamodule_config(self) -> DataModuleSettings:
        """Return the DATAMODULE configuration section.

        Returns:
            DataModuleSettings instance.

        Raises:
            ValueError: If DATAMODULE section is absent.
        """
        if self.DATAMODULE is None:
            raise ValueError("No DATAMODULE configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        """Return the DATASET configuration section.

        Returns:
            DatasetSettings instance.

        Raises:
            ValueError: If DATASET section is absent.
        """
        if self.DATASET is None:
            raise ValueError("No DATASET configuration available")
        return self.DATASET
