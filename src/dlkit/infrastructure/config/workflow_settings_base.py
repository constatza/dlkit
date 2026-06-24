"""Base workflow settings shared by training and inference flows."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Self

from pydantic import Field, model_validator

from .core.base_settings import BasicSettings
from .core.patching import patch_model
from .core.sources import DLKitTomlSource, _read_env_patches
from .datamodule_settings import DataModuleSettings
from .dataset_settings import DatasetSettings
from .extras_settings import ExtrasSettings
from .model_components import ModelComponentSettings


# ponytail: minimal stub; engine files that use BaseWorkflowSettings still depend on this.
# Removed by Task 3 (engine wiring) when BaseWorkflowSettings is fully retired.
class SessionSettings(BasicSettings):
    """Minimal stub — session_settings.py removed in config redesign."""

    workflow: str = "train"

    @property
    def is_inference_mode(self) -> bool:
        return self.workflow == "inference"


class BaseWorkflowSettings(BasicSettings):
    """Base settings class implementing common functionality for all workflows."""

    SESSION: SessionSettings | None = Field(
        default=None, description="Session mode control and execution settings"
    )
    MODEL: ModelComponentSettings | None = Field(default=None, description="Model configuration")
    DATAMODULE: DataModuleSettings | None = Field(
        default=None,
        description="Data loading and processing configuration",
    )
    DATASET: DatasetSettings | None = Field(
        default=None, description="Dataset-specific configuration"
    )
    EXTRAS: ExtrasSettings | None = Field(
        default=None,
        description="Optional free-form helper options for user scripts; ignored by core",
    )

    _workflow_type: ClassVar[str] = "base"

    @model_validator(mode="after")
    def validate_nested_paths(self) -> BaseWorkflowSettings:
        """Validate nested DATASET feature/target paths eagerly."""
        if self.DATASET is not None:
            from dlkit.infrastructure.config.validators import _validate_entry_paths

            _validate_entry_paths(self.DATASET.features, "Feature")
            _validate_entry_paths(self.DATASET.targets, "Target")
        return self

    @classmethod
    def from_toml(
        cls,
        config_path: Path | str,
        *,
        sections: list[str] | None = None,
        **overrides: str | int | float | bool | None,
    ) -> Self:
        """Load workflow settings from a TOML file."""
        source = DLKitTomlSource(Path(config_path), sections=sections)
        settings: Self = cls.model_validate(source())
        if env := _read_env_patches("DLKIT_", "__"):
            settings = patch_model(settings, env)
        if overrides:
            settings = patch_model(settings, overrides)
        return settings

    @property
    def is_training(self) -> bool:
        return not (self.SESSION and self.SESSION.is_inference_mode)

    @property
    def is_inference(self) -> bool:
        return bool(self.SESSION and self.SESSION.is_inference_mode)

    @property
    def is_testing(self) -> bool:
        return False

    @property
    def has_data_config(self) -> bool:
        return self.DATAMODULE is not None and self.DATASET is not None

    def get_datamodule_config(self) -> DataModuleSettings:
        if not self.DATAMODULE:
            raise ValueError("No datamodule configuration available")
        return self.DATAMODULE

    def get_dataset_config(self) -> DatasetSettings:
        if not self.DATASET:
            raise ValueError("No dataset configuration available")
        return self.DATASET
