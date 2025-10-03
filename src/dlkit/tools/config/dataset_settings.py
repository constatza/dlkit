"""Dataset settings - flattened top-level configuration."""

from __future__ import annotations

from pathlib import Path
from pydantic import Field, NonNegativeFloat, FilePath, DirectoryPath

from .core.base_settings import ComponentSettings, BasicSettings
from .enums import DatasetFamily
from .data_entries import Feature, Target


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


class DatasetSettings(ComponentSettings):
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

    name: str = Field(default="FlexibleDataset", description="Dataset class name")
    module_path: str = Field(
        default="dlkit.core.datasets", description="Module path where the dataset class is located"
    )
    # Optional dataset family hint for strategy selection (accepts strings or StrEnum)
    type: DatasetFamily | None = Field(
        default=None,
        description="Dataset family type hint (flexible, graph, timeseries)",
    )
    root: DirectoryPath | None = Field(
        default=None, description="Root directory of the dataset", alias="root_dir"
    )
    # Flexible entries only: arrays of Feature/Target settings
    features: tuple[Feature, ...] = Field(default=(), description="Flexible feature entries")
    targets: tuple[Target, ...] = Field(default=(), description="Flexible target entries")

    split: IndexSplitSettings = Field(
        default_factory=IndexSplitSettings, description="Index split configuration"
    )

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
            files[f.name] = Path(f.path)
        for t in self.targets:
            files[t.name] = Path(t.path)
        return files

    def get_split_config(self) -> dict:
        """Get split configuration as dictionary.

        Returns:
            dict: Index split configuration
        """
        return self.split.to_dict()
