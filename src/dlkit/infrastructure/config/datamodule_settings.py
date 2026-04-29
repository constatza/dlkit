"""DataModule settings - flattened top-level configuration."""

from __future__ import annotations

from pydantic import Field, field_validator

from .core.base_settings import (
    ComponentSettings,
    HyperParameterSettings,
    validate_module_path_import,
)
from .dataloader_settings import DataloaderSettings
from .enums import DataModuleName


class DataModuleSettings(ComponentSettings, HyperParameterSettings):
    """Top-level DataModule configuration for dataflow loading and processing.

    Flattened from component architecture to top-level for easier access.
    Replaces: settings.SESSION.training.data_pipeline
    New usage: settings.DATAMODULE

    Contains all dataflow loading and splitting configuration without nested build methods.
    Uses factory pattern for object construction via external factories.

    Args:
        component_name: DataModule class name
        module_path: Module path to datamodule
        dataloader: DataLoader configuration
    """

    # Accept either a known StrEnum or any custom string
    name: str | DataModuleName = Field(
        default=DataModuleName.IN_MEMORY,
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="DataModule class name",
    )
    module_path: str | None = Field(
        default=None,
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Optional module path where the datamodule class is located",
    )

    dataloader: DataloaderSettings = Field(
        default_factory=DataloaderSettings, description="DataLoader configuration"
    )

    @field_validator("module_path", mode="after")
    @classmethod
    def _validate_module_path(cls, v: str | None) -> str | None:
        return validate_module_path_import(v)
