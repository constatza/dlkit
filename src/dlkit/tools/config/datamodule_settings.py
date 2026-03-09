"""DataModule settings - flattened top-level configuration."""

from __future__ import annotations

from pydantic import Field

from .core.base_settings import ComponentSettings, HyperParameterSettings
from .enums import DataModuleName
from .dataloader_settings import DataloaderSettings


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
        default=DataModuleName.IN_MEMORY, description="DataModule class name"
    )
    module_path: str | None = Field(
        default="dlkit.core.datamodules",
        description="Module path where the datamodule class is located",
    )

    dataloader: DataloaderSettings = Field(
        default_factory=DataloaderSettings, description="DataLoader configuration"
    )
