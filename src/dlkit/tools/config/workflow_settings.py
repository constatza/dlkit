"""Workflow-specific settings classes following SOLID principles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Self

from pydantic import Field, ValidationInfo, model_validator
from pydantic_settings import SettingsConfigDict

from .components.model_components import ModelComponentSettings
from .core.base_settings import BasicSettings
from .data_entries import PathFeature, PathTarget
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .extras_settings import ExtrasSettings
from .mlflow_settings import MLflowSettings
from .optuna_settings import OptunaSettings
from .paths_settings import PathsSettings
from .session_settings import SessionSettings
from .training_settings import TrainingSettings


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
    MODEL: ModelComponentSettings | None = Field(default=None, description="Model configuration")
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

    @model_validator(mode="after")
    def validate_nested_paths(self, info: ValidationInfo) -> BaseWorkflowSettings:
        """Validate nested DATASET paths with eager validation.

        Pydantic does not automatically propagate validation context to nested models.
        This validator explicitly validates feature/target paths at the top-level,
        ensuring path existence checks for fail-fast error detection.

        Args:
            info: Pydantic validation info (unused, kept for compatibility).

        Returns:
            The validated settings instance.

        Raises:
            ValueError: If any feature/target path is specified but does not exist.
        """
        # Validate DATASET nested paths
        if self.DATASET is not None:
            for feature in self.DATASET.features:
                if (
                    isinstance(feature, PathFeature)
                    and feature.path is not None
                    and not feature.path.exists()
                ):
                    raise ValueError(f"Feature path does not exist: {feature.path}")
            for target in self.DATASET.targets:
                if (
                    isinstance(target, PathTarget)
                    and target.path is not None
                    and not target.path.exists()
                ):
                    raise ValueError(f"Target path does not exist: {target.path}")

        return self

    @classmethod
    def from_toml(
        cls,
        config_path: Path | str,
        *,
        sections: list[str] | None = None,
        **overrides: Any,
    ) -> Self:
        """Load workflow settings from a TOML file.

        Priority: TOML values < env vars (``DLKIT_<SECTION>__<field>``) < ``**overrides``.

        Args:
            config_path: Path to the TOML configuration file.
            sections: Optional list of top-level section names to load.
                When ``None`` all sections present in the file are loaded.
            **overrides: Top-level field overrides applied last (highest priority).

        Returns:
            Validated settings instance of the calling class.

        Raises:
            FileNotFoundError: If the config file does not exist.
            pydantic.ValidationError: If validation fails.
        """
        from dlkit.tools.config.core.patching import patch_model
        from dlkit.tools.config.core.sources import DLKitTomlSource, _read_env_patches
        from dlkit.tools.io.config import _sync_session_root_to_environment

        source = DLKitTomlSource(Path(config_path), sections=sections)
        settings: Self = cls.model_validate(source())

        if env := _read_env_patches("DLKIT_", "__"):
            settings = patch_model(settings, env)
        if overrides:
            settings = patch_model(settings, overrides)

        _sync_session_root_to_environment(settings)
        return settings

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

    Optional sections can be omitted in the initial TOML and injected later
    via programmatic updates.
    """

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
        """Use only init and env sources; disable file/dotenv auto-loading."""
        return (init_settings, env_settings)

    # Training-specific sections (optional at load, validated before build)
    TRAINING: TrainingSettings | None = Field(
        default=None,
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
        return self.MLFLOW is not None

    @property
    def optuna_enabled(self) -> bool:
        """Check if Optuna is enabled and properly configured."""
        return self.OPTUNA is not None and self.OPTUNA.enabled

    @property
    def has_training_config(self) -> bool:
        """Check if training configuration is available."""
        return self.TRAINING is not None

    def get_training_config(self) -> TrainingSettings:
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

    Optional sections can be omitted initially and supplied before execution.
    """

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
        """Use only init and env sources; disable file/dotenv auto-loading."""
        return (init_settings, env_settings)

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
