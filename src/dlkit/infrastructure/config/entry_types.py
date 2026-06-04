"""Concrete DataEntry implementations.

Hierarchy:
    PathBasedEntry  (DataEntry + IPathBased)
        ├── PathFeature
        └── PathTarget
    ValueBasedEntry (DataEntry + IValueBased)
        ├── ValueFeature
        └── ValueTarget
    Latent          (DataEntry + IRuntimeGenerated + IWritable)
    AutoencoderTarget (PathBasedEntry + IWritable + IFeatureReference)
    Prediction      (DataEntry + IRuntimeGenerated + IWritable)
"""

from abc import ABC
from pathlib import Path

import numpy as np
import torch
from pydantic import Field, ValidationInfo, model_validator
from pydantic_settings import SettingsConfigDict

from .entry_base import DataEntry
from .entry_protocols import (
    IFeatureReference,
    IPathBased,
    IRuntimeGenerated,
    IValueBased,
    IWritable,
)

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


# ---------------------------------------------------------------------------
# Path-based concrete types
# ---------------------------------------------------------------------------


class PathFeature(PathBasedEntry):
    """Feature loaded from a file path.

    When ``path`` points to a directory, it must be a valid zarr array pack
    (as identified by ``dlkit.infrastructure.io.packs.detect_format``).

    Notes:
        When the wrapper is configured with ``is_autoencoder=True`` and no
        explicit targets are provided, the pipeline treats feature entries as
        targets automatically.
    """

    @model_validator(mode="after")
    def _validate_dir_is_array_pack(self) -> PathFeature:
        """Verify that any directory ``path`` is a recognised zarr array pack.

        Returns:
            The validated instance.

        Raises:
            ValueError: If ``path`` is a directory that is not a known pack format.
        """
        if self.path is None or not self.path.is_dir():
            return self
        from dlkit.infrastructure.io.packs import detect_format

        detect_format(self.path)  # raises ValueError if not a known pack format
        return self


class PathTarget(PathBasedEntry, IWritable):
    """Target loaded from a file path.

    Attributes:
        write: When True, saves the corresponding prediction during inference.

    Notes:
        The pipeline enforces a strict 1:1 mapping between target names and
        prediction names during training/validation/testing.
    """

    write: bool = Field(
        default=False, description="Save the prediction related to this target during inference"
    )


# ---------------------------------------------------------------------------
# Value-based concrete types
# ---------------------------------------------------------------------------


class ValueFeature(ValueBasedEntry):
    """Feature supplied as an in-memory tensor or array.

    Notes:
        When the wrapper is configured with ``is_autoencoder=True`` and no
        explicit targets are provided, the pipeline treats feature entries as
        targets automatically.
    """


class ValueTarget(ValueBasedEntry, IWritable):
    """Target supplied as an in-memory tensor or array.

    Attributes:
        write: When True, saves the corresponding prediction during inference.

    Notes:
        The pipeline enforces a strict 1:1 mapping between target names and
        prediction names during training/validation/testing.
    """

    write: bool = Field(
        default=False, description="Save the prediction related to this target during inference"
    )


# ---------------------------------------------------------------------------
# Runtime / special entry types
# ---------------------------------------------------------------------------


class Latent(DataEntry, IRuntimeGenerated, IWritable):
    """Intermediate representation generated by the model at run-time.

    Latents are not handled by ``FlexibleDataset``; the processing pipeline
    manages them during inference.

    Attributes:
        write: When True, persists this latent during inference.
    """

    write: bool = Field(default=False, description="Save this latent during inference")

    def has_value(self) -> bool:
        """Latents have no in-memory value — they are generated at run-time.

        Returns:
            Always False.
        """
        return False

    def has_path(self) -> bool:
        """Latents have no file path.

        Returns:
            Always False.
        """
        return False

    def is_placeholder(self) -> bool:
        """Latents are never placeholders.

        Returns:
            Always False.
        """
        return False


class AutoencoderTarget(PathBasedEntry, IWritable, IFeatureReference):
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

    _resolved_value: torch.Tensor | np.ndarray | None = None

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


class Prediction(DataEntry, IRuntimeGenerated, IWritable):
    """Model output entry corresponding to a specific target.

    Attributes:
        target_name: Name of the target this prediction is paired with.
        write: When True, saves prediction data during inference.
    """

    target_name: str = Field(..., description="Corresponding target name")
    write: bool = Field(default=True, description="Save predictions during inference")

    def has_value(self) -> bool:
        """Predictions are generated at run-time and have no in-memory value.

        Returns:
            Always False.
        """
        return False

    def has_path(self) -> bool:
        """Predictions have no file path.

        Returns:
            Always False.
        """
        return False

    def is_placeholder(self) -> bool:
        """Predictions are never placeholders.

        Returns:
            Always False.
        """
        return False
