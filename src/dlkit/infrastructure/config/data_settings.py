"""Data settings — unified dataset, dataloader, and DataModule configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field
from pydantic.types import NonNegativeInt, PositiveInt

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.data_entries import AnyEntry
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.split_settings import IndexSplitSettings


class DataModuleSelector(BasicSettings):
    """Lightning DataModule class selector (replaces DATAMODULE.name/module_path).

    Args:
        name: DataModule class name (alias: class).
        module_path: Python module path where the class is defined.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(default="InMemoryModule", alias="class")
    module_path: str | None = None


class DataSettings(BasicSettings):
    """Unified data configuration: dataset class, loader settings, splits, and entries.

    Merges the old DATASET and DATAMODULE sections. DataLoader fields are promoted
    to the top level; the Lightning DataModule class lives in the data.module sub-table.

    Args:
        name: Dataset class name (alias: class).
        module_path: Python module path for the dataset class.
        family: Dataset family (flexible, graph).
        root: Root directory for data files.
        batch_size: Number of samples per batch.
        num_workers: Number of DataLoader worker processes.
        shuffle: Whether to shuffle training data each epoch.
        pin_memory: Whether to pin memory in DataLoader.
        persistent_workers: Whether to keep DataLoader workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        follow_batch: Graph dataset keys that need follow_batch treatment.
        features: Feature entry configurations.
        targets: Target entry configurations.
        splits: Train/val/test split configuration.
        module: Lightning DataModule class selector.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    # Dataset class selector
    name: str | None = Field(default=None, alias="class")
    module_path: str | None = None
    family: DatasetFamily | None = None
    root: Path | None = None
    # DataLoader settings (promoted from DATAMODULE.dataloader)
    batch_size: PositiveInt = 64
    num_workers: NonNegativeInt = 0
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: PositiveInt | None = None
    follow_batch: tuple[str, ...] | None = None
    # Entries
    features: tuple[AnyEntry, ...] = ()
    targets: tuple[AnyEntry, ...] = ()
    # Split config (was DATAMODULE.split)
    splits: IndexSplitSettings = Field(default_factory=IndexSplitSettings)
    # DataModule class selector (was DATAMODULE.name)
    module: DataModuleSelector = Field(default_factory=DataModuleSelector)
