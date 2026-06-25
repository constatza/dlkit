"""Data settings — unified dataset, dataloader, and DataModule configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, ConfigDict, Field, field_validator, model_validator
from pydantic.types import NonNegativeInt, PositiveInt

from dlkit.infrastructure.config.core.base_settings import BasicSettings
from dlkit.infrastructure.config.data_entries import AnyEntry
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.enums import DatasetFamily
from dlkit.infrastructure.config.split_settings import IndexSplitSettings

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".npy": "npy",
    ".npz": "npz",
    ".csv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".zarr": "zarr",
}

type RawEntryAtom = str | int | float | bool | Path | None
type RawEntrySequence = list[RawEntryAtom] | tuple[RawEntryAtom, ...]
type RawEntryValue = RawEntryAtom | RawEntrySequence
type RawEntryPayload = dict[str, RawEntryValue]
type RawEntryItem = RawEntryPayload | AnyEntry
type RawEntryCollection = list[RawEntryItem] | tuple[RawEntryItem, ...]
type RawEntryFallback = RawEntryAtom | RawEntryPayload


def _as_raw_entry_payload(entry: RawEntryItem) -> RawEntryPayload | None:
    """Return raw TOML payload dict, or None for already-validated entry objects."""
    if isinstance(entry, dict):
        return entry
    return None


def _infer_entry_format(entry: RawEntryItem) -> RawEntryItem:
    """Fill in missing path-entry format from file extension.

    Runs before Pydantic resolves the AnyEntry discriminated union so
    path-based entries can omit ``format`` in TOML and still parse correctly.

    Args:
        entry: Raw entry payload dict or validated AnyEntry.

    Returns:
        Entry with ``format`` injected, or original when no inference is needed.

    Raises:
        ValueError: If the path extension is unknown and format cannot be inferred.
    """
    payload = _as_raw_entry_payload(entry)
    if payload is None:
        return entry
    if "format" in payload:
        return entry
    if "feature_ref" in payload or "value" in payload:
        return entry

    raw_path = payload.get("path")
    if raw_path is None:
        return entry
    if not isinstance(raw_path, str | Path):
        return entry

    path = Path(raw_path)
    suffix = path.suffix.lower()
    inferred = _FORMAT_BY_SUFFIX.get(suffix)
    if inferred is not None:
        return {**payload, "format": inferred}

    if path.exists() and path.is_dir():
        raise ValueError(
            f"Could not infer data entry format from directory path '{path}'. "
            "Directory-backed entries must use a '.zarr' suffix for automatic inference, "
            "or specify an explicit format such as format = 'zarr'."
        )

    raise ValueError(
        f"Could not infer data entry format from path '{path}'. "
        "Supported inferred extensions are: .npy, .npz, .csv, .txt, .parquet, .h5, .hdf5, .zarr. "
        "Add an explicit format = '...' if the path uses a different convention."
    )


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
        root: Root directory for data files (also accepts root_dir alias).
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
    # exclude=True: root is resolved at runtime via ds_overrides, not passed to dataset __init__
    root: Path | None = Field(
        default=None,
        exclude=True,
        validation_alias=AliasChoices("root", "root_dir"),
    )
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

    @field_validator("features", "targets", mode="before")
    @classmethod
    def _inject_missing_entry_formats(
        cls, value: RawEntryCollection | RawEntryFallback
    ) -> RawEntryCollection | RawEntryFallback:
        """Infer missing ``format`` discriminators for raw dataset entry payloads."""
        if isinstance(value, list):
            return [_infer_entry_format(entry) for entry in value]
        if isinstance(value, tuple):
            return tuple(_infer_entry_format(entry) for entry in value)
        return value

    @field_validator("features", mode="after")
    @classmethod
    def _force_feature_role(cls, values: tuple[AnyEntry, ...]) -> tuple[AnyEntry, ...]:
        """Ensure all feature entries carry the FEATURE data role."""
        return tuple(
            v
            if v.data_role == DataRole.FEATURE
            else v.model_copy(update={"data_role": DataRole.FEATURE})
            for v in values
        )

    @field_validator("targets", mode="after")
    @classmethod
    def _force_target_role(cls, values: tuple[AnyEntry, ...]) -> tuple[AnyEntry, ...]:
        """Ensure all target entries carry the TARGET data role."""
        return tuple(
            v
            if v.data_role == DataRole.TARGET
            else v.model_copy(update={"data_role": DataRole.TARGET})
            for v in values
        )

    @model_validator(mode="after")
    def _validate_nested_paths(self) -> DataSettings:
        """Validate nested feature/target paths eagerly."""
        from dlkit.infrastructure.config.validators import _validate_entry_paths

        _validate_entry_paths(self.features, "Feature")
        _validate_entry_paths(self.targets, "Target")
        return self
