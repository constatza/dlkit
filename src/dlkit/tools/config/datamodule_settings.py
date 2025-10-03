"""DataModule settings - flattened top-level configuration."""

from __future__ import annotations

from lightning import LightningDataModule
from pydantic import Field

from dlkit.core.datatypes.base import IntHyperparameter
from .core.base_settings import ComponentSettings, HyperParameterSettings
from .enums import DataModuleName
from .dataloader_settings import DataloaderSettings


class DataModuleSettings(ComponentSettings[LightningDataModule], HyperParameterSettings):
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
        batch_size: Override batch size for convenience
        num_workers: Override number of workers for convenience
    """

    # Accept either a known StrEnum or any custom string
    name: str | DataModuleName = Field(
        default=DataModuleName.IN_MEMORY, description="DataModule class name"
    )
    module_path: str = Field(
        default="dlkit.core.datamodules",
        description="Module path where the datamodule class is located",
    )

    dataloader: DataloaderSettings = Field(
        default_factory=DataloaderSettings, description="DataLoader configuration"
    )

    # Convenience fields for common overrides
    batch_size: IntHyperparameter | None = Field(
        default=None, description="Batch size override (if None, uses dataloader.batch_size)"
    )
    num_workers: int | None = Field(
        default=None,
        description="Number of workers override (if None, uses dataloader.num_workers)",
    )

    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size with override support.

        Returns:
            int: Effective batch size to use
        """
        batch_size = self.batch_size
        if batch_size is not None:
            # Handle hyperparameter types
            if isinstance(batch_size, (int, float)):
                return int(batch_size)
            elif isinstance(batch_size, dict):
                # Handle hyperparameter dict specifications
                if 'default' in batch_size:
                    default_val = batch_size['default']
                    if isinstance(default_val, (int, float)):
                        return int(default_val)
                    elif isinstance(default_val, tuple) and default_val:
                        return int(default_val[0])  # Use first element
                    else:
                        return 32
                elif 'min' in batch_size and 'max' in batch_size:
                    min_val = batch_size['min']
                    if isinstance(min_val, (int, float)):
                        return int(min_val)
                    elif isinstance(min_val, tuple) and min_val:
                        return int(min_val[0])  # Use first element
                    else:
                        return 32
                else:
                    return 32  # Fallback
            elif hasattr(batch_size, '__int__'):
                try:
                    return int(batch_size)
                except (ValueError, TypeError):
                    return 32
            else:
                # For hyperparameter objects, try to get their value
                value = getattr(batch_size, 'value', getattr(batch_size, 'default', 32))
                if isinstance(value, (int, float)):
                    return int(value)
                # Fallback for complex hyperparameter types
                return 32
        return getattr(self.dataloader, "batch_size", 32)

    @property
    def effective_num_workers(self) -> int:
        """Get effective number of workers with override support.

        Returns:
            int: Effective number of workers to use
        """
        return (
            self.num_workers
            if self.num_workers is not None
            else getattr(self.dataloader, "num_workers", 4)
        )

    def get_dataloader_config(self) -> dict:
        """Get dataloader configuration with overrides applied.

        Returns:
            dict: DataLoader initialization kwargs with effective values
        """
        config = self.dataloader.to_dict()

        # Apply overrides
        if self.batch_size is not None:
            config["batch_size"] = self.batch_size
        if self.num_workers is not None:
            config["num_workers"] = self.num_workers

        return config
