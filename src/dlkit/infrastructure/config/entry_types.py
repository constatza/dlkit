"""Concrete DataEntry implementations.

Hierarchy:
    PathBasedEntry  (DataEntry + IPathBased)
        ├── ZarrEntry
        ├── NpyEntry
        ├── NpzEntry
        ├── CsvEntry
        ├── ParquetEntry
        └── Hdf5Entry
    ValueBasedEntry (DataEntry + IValueBased)
        └── ValueEntry
    Latent          (DataEntry + IRuntimeGenerated)
    AutoencoderTarget (PathBasedEntry + IFeatureReference)
    Prediction      (DataEntry + IRuntimeGenerated)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import Field, PrivateAttr, ValidationInfo, model_validator
from pydantic_settings import SettingsConfigDict

from dlkit.common.sources import ArraySource
from dlkit.infrastructure.zarr import ZarrLazyReader

from .data_roles import DataRole
from .entry_base import DataEntry
from .entry_protocols import (
    IFeatureReference,
    IPathBased,
    IRuntimeGenerated,
    IValueBased,
)
from .normalized_entry import NormalizedEntry

# ---------------------------------------------------------------------------
# Abstract path-based and value-based bases
# ---------------------------------------------------------------------------


class PathBasedEntry(DataEntry, IPathBased, ABC):
    """Base for entries that load data from a file path.

    Attributes:
        path: Path to the data file; None enables placeholder mode.
    """

    path: Path | None = Field(
        default=None, description="Path to the data file (None for placeholder mode)"
    )

    def get_path(self) -> Path | None:
        """Return the file path.

        Returns:
            Path if set, else None.
        """
        return self.path

    def has_value(self) -> bool:
        """Path-based entries never hold an in-memory value.

        Returns:
            Always False.
        """
        return False

    def has_path(self) -> bool:
        """Return True if a path is set.

        Returns:
            True when ``self.path`` is not None.
        """
        return self.path is not None

    def is_placeholder(self) -> bool:
        """Return True when no path has been provided yet.

        Returns:
            True if ``self.path`` is None.
        """
        return self.path is None

    @model_validator(mode="after")
    def validate_path_existence(self, info: ValidationInfo) -> PathBasedEntry:
        """Fail fast when a specified path does not exist on disk.

        Args:
            info: Pydantic validation context (unused).

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``path`` is set but does not exist.
        """
        if self.path is not None and not self.path.exists():
            raise ValueError(f"Path does not exist: {self.path}")
        return self

    @abstractmethod
    def open_reader(self) -> ArraySource | Path:
        """Return the IO source for this entry.

        Returns:
            ``ArraySource`` for lazy formats (zarr) — callers iterate samples
            via ``get_item()`` / ``get_batch()``.  ``Path`` for all eager formats
            — callers load the whole array via ``load_array()``.
        """

    @property
    def load_kwargs(self) -> dict[str, Any]:
        """Extra kwargs forwarded to load_array() when source is a Path.

        Returns:
            Empty dict by default; overridden by NpyEntry, NpzEntry.
        """
        return {}

    @property
    def array_key(self) -> str:
        """Array key used when loading multi-array sources.

        Returns:
            Entry name by default; overridden by NpzEntry.
        """
        return self.name or ""

    def normalize(self) -> NormalizedEntry:
        """Return a NormalizedEntry with the path-based source.

        Returns:
            NormalizedEntry wrapping the open_reader() result.

        Raises:
            PlaceholderNotResolvedError: If path is None.
        """
        if self.is_placeholder():
            from dlkit.common.errors import PlaceholderNotResolvedError

            raise PlaceholderNotResolvedError(str(self.name or "unknown"))
        return NormalizedEntry(
            source=self.open_reader(),
            array_key=self.array_key,
            load_kwargs=self.load_kwargs,
        )


class ValueBasedEntry(DataEntry, IValueBased, ABC):
    """Base for entries that receive data programmatically.

    Attributes:
        value: In-memory tensor or numpy array; None enables placeholder mode.
    """

    value: torch.Tensor | np.ndarray | None = Field(
        default=None,
        description="In-memory tensor/array (None for placeholder mode)",
        exclude=True,
    )

    def get_value(self) -> torch.Tensor | np.ndarray | None:
        """Return the in-memory value.

        Returns:
            Tensor or array if set, else None.
        """
        return self.value

    def has_value(self) -> bool:
        """Return True when a value has been provided.

        Returns:
            True if ``self.value`` is not None.
        """
        return self.value is not None

    def has_path(self) -> bool:
        """Value-based entries never have a file path.

        Returns:
            Always False.
        """
        return False

    def is_placeholder(self) -> bool:
        """Return True when no value has been provided yet.

        Returns:
            True if ``self.value`` is None.
        """
        return self.value is None

    def normalize(self) -> NormalizedEntry:
        """Return a NormalizedEntry with the in-memory value as source.

        Returns:
            NormalizedEntry wrapping self.value.

        Raises:
            PlaceholderNotResolvedError: If value is None.
        """
        if self.is_placeholder():
            from dlkit.common.errors import PlaceholderNotResolvedError

            raise PlaceholderNotResolvedError(str(self.name or "unknown"))
        assert self.name is not None
        return NormalizedEntry(
            source=self.value,  # ty: ignore[invalid-argument-type]
            array_key=self.name,
        )


# ---------------------------------------------------------------------------
# Runtime / special entry types
# ---------------------------------------------------------------------------


class Latent(DataEntry, IRuntimeGenerated):
    """Intermediate representation generated by the model at run-time.

    Latents are not handled by ``FlexibleDataset``; the processing pipeline
    manages them during inference.

    Attributes:
        write: When True, persists this latent during inference.
    """

    data_role: DataRole = DataRole.LATENT
    write: bool = Field(default=False, description="Save this latent during inference")

    def normalize(self) -> NormalizedEntry:
        """Latents are runtime-generated and cannot be normalized.

        Raises:
            TypeError: Always — latents are not valid FlexibleDataset inputs.
        """
        raise TypeError(
            f"Latent entry '{self.name}' is runtime-generated and cannot be normalized. "
            "Latents are not inputs to FlexibleDataset."
        )


class AutoencoderTarget(PathBasedEntry, IFeatureReference):
    """Reconstruction target that derives its transform chain from a feature.

    ``AutoencoderTarget`` automatically builds an inverted transform chain by
    reversing the referenced feature's transforms and calling
    ``inverse_transform()`` on each.  This is essential for stateful
    transforms such as ``SampleNormL2`` that cache per-sample state during the
    forward pass.

    Attributes:
        feature_ref: Name of the ``Feature`` entry to reference.
        write: When True, saves reconstruction data during inference.

    Example:
        ::

            x = Feature(
                name="x",
                path="input.npy",
                transforms=[TransformSettings(name="SampleNormL2")],
            )
            y = AutoencoderTarget(name="y", feature_ref="x")
    """

    feature_ref: str = Field(
        ..., description="Name of the Feature entry to reference for transform inversion"
    )
    write: bool = Field(default=False, description="Save reconstruction data during inference")

    _resolved_value: torch.Tensor | np.ndarray | None = PrivateAttr(default=None)

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_feature_ref(self) -> AutoencoderTarget:
        """Require ``feature_ref`` and warn when manual transforms are provided.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``feature_ref`` is empty.
        """
        if not self.feature_ref:
            raise ValueError("AutoencoderTarget must have 'feature_ref' specified")

        if self.transforms:
            from dlkit.infrastructure.utils.logging_config import get_logger

            get_logger(__name__).warning(
                "AutoencoderTarget '{}' ignores manual transforms because feature_ref='{}' "
                "derives them automatically.",
                self.name,
                self.feature_ref,
            )

        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the data file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path

    def has_value(self) -> bool:
        """Return True when the feature reference has been resolved to a value.

        Returns:
            True if ``_resolved_value`` is set.
        """
        return self._resolved_value is not None

    def is_placeholder(self) -> bool:
        """Return True until both path and resolved value are unavailable.

        Returns:
            True if neither ``path`` nor ``_resolved_value`` is set.
        """
        return self.path is None and self._resolved_value is None


class Prediction(DataEntry, IRuntimeGenerated):
    """Model output entry corresponding to a specific target.

    Attributes:
        target_name: Name of the target this prediction is paired with.
        write: When True, saves prediction data during inference.
    """

    data_role: DataRole = DataRole.TARGET
    target_name: str = Field(..., description="Corresponding target name")
    write: bool = Field(default=True, description="Save predictions during inference")

    def normalize(self) -> NormalizedEntry:
        """Predictions are runtime-generated and cannot be normalized.

        Raises:
            TypeError: Always — predictions are not valid FlexibleDataset inputs.
        """
        raise TypeError(
            f"Prediction entry '{self.name}' is runtime-generated and cannot be normalized. "
            "Predictions are not inputs to FlexibleDataset."
        )


# ---------------------------------------------------------------------------
# Format-specific path-based entry types
# ---------------------------------------------------------------------------


class ZarrEntry(PathBasedEntry):
    """Feature or target stored as a native zarr array."""

    format: Literal["zarr"] = "zarr"
    chunk_size: int = Field(default=1, gt=0)

    @model_validator(mode="after")
    def _validate_zarr_store(self) -> ZarrEntry:
        """Verify that any directory path is a valid zarr v3 store.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the directory is missing zarr.json.
        """
        if self.path is None or not self.path.is_dir():
            return self
        if not (self.path / "zarr.json").exists():
            raise ValueError(f"Not a native zarr store (missing zarr.json): {self.path}")
        return self

    def open_reader(self) -> ArraySource:
        """Return a lazy reader for the zarr array.

        Returns:
            ``ZarrLazyReader`` wrapping the native zarr array at ``self.path``,
            which satisfies the ``ArraySource`` protocol.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return ZarrLazyReader(self.path)


class NpyEntry(PathBasedEntry):
    """Feature or target stored as a NumPy .npy file."""

    format: Literal["npy"] = "npy"
    mmap: bool = Field(default=True, description="Memory-map the file (zero-copy, OOM-safe)")

    @model_validator(mode="after")
    def _validate_npy_suffix(self) -> NpyEntry:
        """Require a .npy file extension.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the path does not have a .npy suffix.
        """
        if self.path is not None and self.path.suffix != ".npy":
            raise ValueError(f"NpyEntry requires a .npy file, got: {self.path}")
        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the .npy file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path

    @property
    def load_kwargs(self) -> dict[str, Any]:
        """Extra kwargs forwarded to load_array().

        Returns:
            Dict with mmap_mode when memory-mapping is enabled.
        """
        return {"mmap_mode": "r"} if self.mmap else {}


class NpzEntry(PathBasedEntry):
    """Feature or target stored as a NumPy .npz archive."""

    format: Literal["npz"] = "npz"
    mmap: bool = Field(default=True)
    key: str | None = Field(
        default=None,
        description="Array key within the archive; None uses entry name",
    )

    @model_validator(mode="after")
    def _validate_npz_suffix(self) -> NpzEntry:
        """Require a .npz file extension.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the path does not have a .npz suffix.
        """
        if self.path is not None and self.path.suffix != ".npz":
            raise ValueError(f"NpzEntry requires a .npz file, got: {self.path}")
        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the .npz file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path

    @property
    def load_kwargs(self) -> dict[str, Any]:
        """Extra kwargs forwarded to load_array().

        Returns:
            Dict with mmap_mode when memory-mapping is enabled.
        """
        return {"mmap_mode": "r"} if self.mmap else {}

    @property
    def array_key(self) -> str:
        """Array key within the .npz archive.

        Returns:
            Explicit key if set, otherwise the entry name.
        """
        return self.key or self.name or ""


class CsvEntry(PathBasedEntry):
    """Feature or target stored as a CSV or text file."""

    format: Literal["csv"] = "csv"
    delimiter: str = ","
    header: bool = True

    @model_validator(mode="after")
    def _validate_csv_suffix(self) -> CsvEntry:
        """Require a .csv or .txt file extension.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the path suffix is not .csv or .txt.
        """
        if self.path is not None and self.path.suffix not in {".csv", ".txt"}:
            raise ValueError(f"CsvEntry requires .csv/.txt, got: {self.path}")
        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the CSV file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path


class ParquetEntry(PathBasedEntry):
    """Feature or target stored as a Parquet file."""

    format: Literal["parquet"] = "parquet"

    @model_validator(mode="after")
    def _validate_parquet_suffix(self) -> ParquetEntry:
        """Require a .parquet file extension.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the path suffix is not .parquet.
        """
        if self.path is not None and self.path.suffix != ".parquet":
            raise ValueError(f"ParquetEntry requires .parquet, got: {self.path}")
        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the Parquet file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path


class Hdf5Entry(PathBasedEntry):
    """Feature or target stored in an HDF5 file."""

    format: Literal["hdf5"] = "hdf5"
    dataset_key: str = Field(
        default="data",
        description="HDF5 dataset path within the file",
    )

    @model_validator(mode="after")
    def _validate_hdf5_suffix(self) -> Hdf5Entry:
        """Require a .h5 or .hdf5 file extension.

        Returns:
            The validated instance.

        Raises:
            ValueError: If the path suffix is not .h5 or .hdf5.
        """
        if self.path is not None and self.path.suffix not in {".h5", ".hdf5"}:
            raise ValueError(f"Hdf5Entry requires .h5/.hdf5, got: {self.path}")
        return self

    def open_reader(self) -> Path:
        """Return the file path as the IO source.

        Returns:
            The path to the HDF5 file.
        """
        if self.path is None:
            raise ValueError(f"{type(self).__name__} has no path — call is_placeholder() first")
        return self.path


# ---------------------------------------------------------------------------
# Unified value-based entry type
# ---------------------------------------------------------------------------


class ValueEntry(ValueBasedEntry):
    """Feature or target supplied as an in-memory tensor or array.

    Replaces the former per-role value entry classes. The write field (from DataEntry)
    controls whether predictions are written during inference.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Abstract bases
    "PathBasedEntry",
    "ValueBasedEntry",
    # Format-specific path-based types
    "ZarrEntry",
    "NpyEntry",
    "NpzEntry",
    "CsvEntry",
    "ParquetEntry",
    "Hdf5Entry",
    # Unified value-based type
    "ValueEntry",
    # Runtime / special types
    "Latent",
    "AutoencoderTarget",
    "Prediction",
]
