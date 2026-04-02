"""Dataset settings - flattened top-level configuration."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    NonNegativeFloat,
    ValidationInfo,
    model_validator,
)

from .core.base_settings import BasicSettings, StringNamedComponentSettings
from .data_entries import (
    FeatureType,
    PathFeature,
    PathTarget,
    TargetType,
)
from .enums import DatasetFamily


class IndexSplitSettings(BasicSettings):
    """Index split configuration for train/val/test dataflow splitting.

    Pure configuration without build methods - uses factory pattern.

    Args:
        filepath: Path to existing index split file
        test_ratio: Fraction of dataflow for testing
        val_ratio: Fraction of dataflow for validation
    """

    filepath: FilePath | None = Field(default=None, description="Path to index split file")
    test_ratio: NonNegativeFloat = Field(
        default=0.15, description="Fraction of dataflow used for testing", alias="test"
    )
    val_ratio: NonNegativeFloat = Field(
        default=0.15, description="Fraction of dataflow used for validation", alias="val"
    )

    @property
    def has_existing_split(self) -> bool:
        """Check if existing split file is configured.

        Returns:
            bool: True if filepath is specified
        """
        return self.filepath is not None

    @property
    def train_ratio(self) -> float:
        """Calculate training ratio.

        Returns:
            float: Fraction of dataflow for training
        """
        return 1.0 - self.test_ratio - self.val_ratio

    def get_split_ratios(self) -> tuple[float, float, float]:
        """Get all split ratios.

        Returns:
            tuple[float, float, float]: (train, val, test) ratios
        """
        return (self.train_ratio, self.val_ratio, self.test_ratio)


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
        x: Features file path
        y: Targets file path
        split: Index split configuration
    """

    name: str | Callable[..., Any] | dict[str, Any] | None = Field(
        default="FlexibleDataset", description="Dataset class name"
    )
    module_path: str | None = Field(
        default=None, description="Optional module path where the dataset class is located"
    )
    # Optional dataset family hint for strategy selection (accepts strings or StrEnum)
    family: DatasetFamily | None = Field(
        default=None,
        description="Explicit dataset family (flexible, graph, timeseries)",
    )
    type: DatasetFamily | None = Field(
        default=None,
        description="Dataset family type hint (flexible, graph, timeseries)",
    )
    root: DirectoryPath | None = Field(
        default=None, description="Root directory of the dataset", alias="root_dir"
    )
    # Flexible entries only: tuples of Feature/Target settings (immutable for consistency)
    features: tuple[FeatureType, ...] = Field(default=(), description="Flexible feature entries")
    targets: tuple[TargetType, ...] = Field(default=(), description="Flexible target entries")

    split: IndexSplitSettings = Field(
        default_factory=IndexSplitSettings, description="Index split configuration"
    )
    memmap_cache: bool = Field(
        default=False,
        description=(
            "If true, load dataset using OS memory-mapped files. "
            "Enables out-of-memory datasets that do not fit in RAM. "
            "All entries must be file-backed (PathBasedEntry). "
            "Cache stored at platformdirs.user_cache_path('dlkit') / 'memmap'. "
            "Auto-invalidates when source files or dtype change."
        ),
    )

    @model_validator(mode="after")
    def validate_nested_paths(self, info: ValidationInfo) -> DatasetSettings:
        """Validate nested Feature/Target paths with eager validation.

        Pydantic does not automatically propagate validation context to nested models.
        This validator explicitly validates features and targets paths, ensuring
        path existence checks are performed for fail-fast error detection.

        Args:
            info: Pydantic validation info (unused, kept for compatibility).

        Returns:
            The validated DatasetSettings instance.

        Raises:
            ValueError: If any feature/target path is specified but does not exist.
        """
        # Validate features
        for feature in self.features:
            if (
                isinstance(feature, PathFeature)
                and feature.path is not None
                and not feature.path.exists()
            ):
                raise ValueError(f"Feature path does not exist: {feature.path}")

        # Validate targets
        for target in self.targets:
            if (
                isinstance(target, PathTarget)
                and target.path is not None
                and not target.path.exists()
            ):
                raise ValueError(f"Target path does not exist: {target.path}")

        return self

    @property
    def resolved_memmap_cache_dir(self) -> Path | None:
        """Resolve the OS-standard memmap cache directory when enabled.

        Returns:
            Path | None: Cache directory path, or None when memmap_cache is False.
        """
        if not self.memmap_cache:
            return None
        from platformdirs import user_cache_path

        return user_cache_path("dlkit") / "memmap"

    @property
    def has_targets(self) -> bool:
        """Check if any target entries are configured."""
        try:
            return len(self.targets) > 0
        except Exception:
            return False

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
            if isinstance(f, PathFeature) and f.name is not None and f.path is not None:
                files[f.name] = Path(f.path)
        for t in self.targets:
            if isinstance(t, PathTarget) and t.name is not None and t.path is not None:
                files[t.name] = Path(t.path)
        return files

    def get_split_config(self) -> dict:
        """Get split configuration as dictionary.

        Returns:
            dict: Index split configuration
        """
        return self.split.to_dict()

    def get_init_kwargs(self, exclude: set[str] | None = None) -> dict[str, Any]:
        """Return initialization kwargs preserving nested DataEntry objects."""
        # Exclude memmap_cache bool — FlexibleDataset uses memmap_cache_dir (resolved Path)
        base = super().get_init_kwargs(exclude={"memmap_cache", *(exclude or set())})
        # Preserve DataEntry instances instead of serialized dicts
        base["features"] = list(self.features)
        base["targets"] = list(self.targets)
        base["split"] = self.split
        resolved = self.resolved_memmap_cache_dir
        if resolved is not None:
            base["memmap_cache_dir"] = resolved
        return base
