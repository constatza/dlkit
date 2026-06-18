"""Dataset settings - flattened top-level configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import (
    DirectoryPath,
    Field,
    field_validator,
    model_validator,
)

from .core.base_settings import (
    StringNamedComponentSettings,
    validate_module_path_import,
)
from .data_entries import AnyEntry, PathBasedEntry
from .data_roles import DataRole
from .enums import DatasetFamily
from .split_settings import IndexSplitSettings as IndexSplitSettings

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
    """Return raw TOML payloads while leaving validated entries untouched."""
    if isinstance(entry, dict):
        return entry
    return None


def _infer_entry_format(entry: RawEntryItem) -> RawEntryItem:
    """Fill in missing path-entry format from the raw path payload.

    This runs before Pydantic resolves the ``AnyEntry`` discriminated union so
    path-based entries can omit ``format`` in TOML and still parse into the
    correct concrete entry class.
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
            f"Could not infer DATASET entry format from directory path '{path}'. "
            "Directory-backed entries must use a '.zarr' suffix for automatic inference, "
            "or specify an explicit format such as format = 'zarr'."
        )

    raise ValueError(
        f"Could not infer DATASET entry format from path '{path}'. "
        "Supported inferred extensions are: .npy, .npz, .csv, .txt, .parquet, .h5, .hdf5, .zarr. "
        "Add an explicit format = '...' if the path uses a different convention."
    )


class DatasetSettings(StringNamedComponentSettings):
    """Top-level Dataset configuration.

    Flattened from component architecture to top-level for easier access.
    Replaces: settings.SESSION.training.data_pipeline.dataset
    New usage: settings.DATASET

    Pure configuration without build methods - uses factory pattern.

    Args:
        component_name: Dataset class name
        module_path: Module path to dataset
        root: Root directory of the dataset
        features: Feature entries
        targets: Target entries
    """

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="FlexibleDataset",
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Dataset class name",
    )
    module_path: str | None = Field(
        default=None,
        exclude=True,
        json_schema_extra={"dlkit_init_kwarg": False},
        description="Optional module path where the dataset class is located",
    )
    # Meta-fields: used for routing/path resolution, never forwarded to the dataset constructor.
    family: DatasetFamily | None = Field(
        default=None,
        exclude=True,
        description="Explicit dataset family (flexible, graph, timeseries)",
    )
    type: DatasetFamily | None = Field(
        default=None,
        exclude=True,
        description="Dataset family type hint (flexible, graph, timeseries)",
    )
    root: DirectoryPath | None = Field(
        default=None, exclude=True, description="Root directory of the dataset", alias="root_dir"
    )
    # Flexible entries only: tuples of DataEntry settings (immutable for consistency)
    features: tuple[AnyEntry, ...] = Field(default=(), description="Flexible feature entries")
    targets: tuple[AnyEntry, ...] = Field(default=(), description="Flexible target entries")

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
        """Ensure all feature entries have the FEATURE data role."""
        return tuple(
            v
            if v.data_role == DataRole.FEATURE
            else v.model_copy(update={"data_role": DataRole.FEATURE})
            for v in values
        )

    @field_validator("targets", mode="after")
    @classmethod
    def _force_target_role(cls, values: tuple[AnyEntry, ...]) -> tuple[AnyEntry, ...]:
        """Ensure all target entries have the TARGET data role."""
        return tuple(
            v
            if v.data_role == DataRole.TARGET
            else v.model_copy(update={"data_role": DataRole.TARGET})
            for v in values
        )

    @model_validator(mode="after")
    def validate_nested_paths(self) -> DatasetSettings:
        """Validate nested Feature/Target paths with eager validation."""
        from dlkit.infrastructure.config.validators import _validate_entry_paths

        _validate_entry_paths(self.features, "Feature")
        _validate_entry_paths(self.targets, "Target")
        return self

    @field_validator("module_path", mode="after")
    @classmethod
    def _validate_module_path(cls, v: str | None) -> str | None:
        return validate_module_path_import(v)

    @property
    def has_targets(self) -> bool:
        """Check if any target entries are configured."""
        return len(self.targets) > 0

    @property
    def has_root(self) -> bool:
        """Check if root directory is configured.

        Returns:
            bool: True if root directory is specified
        """
        return self.root is not None

    def get_data_files(self) -> dict[str, Path | None]:
        """Get dataflow file paths.

        Returns:
            dict[str, Path | None]: Dictionary with x, y file paths
        """
        files: dict[str, Path | None] = {}
        for f in self.features:
            if isinstance(f, PathBasedEntry) and f.name is not None and f.path is not None:
                files[f.name] = Path(f.path)
        for t in self.targets:
            if isinstance(t, PathBasedEntry) and t.name is not None and t.path is not None:
                files[t.name] = Path(t.path)
        return files

    def get_init_kwargs(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Return initialization kwargs preserving nested DataEntry objects."""
        base = super().get_init_kwargs(exclude=exclude)
        # Remove legacy features/targets keys — FlexibleDataset uses entries= only.
        base.pop("features", None)
        base.pop("targets", None)
        # Combine features and targets into a single entries list.
        # Graph/timeseries datasets don't accept entries= so we omit when empty.
        entries = list(self.features) + list(self.targets)
        if entries:
            base["entries"] = entries
        return base
