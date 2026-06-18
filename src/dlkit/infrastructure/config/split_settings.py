"""Index split configuration."""

from __future__ import annotations

from pydantic import Field, FilePath, NonNegativeFloat

from .core.base_settings import BasicSettings


class IndexSplitSettings(BasicSettings):
    """Index split configuration for train/val/test dataflow splitting.

    Args:
        filepath: Path to existing index split file.
        test_ratio: Fraction of dataflow used for testing.
        val_ratio: Fraction of dataflow used for validation.
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
            True if filepath is specified.
        """
        return self.filepath is not None

    @property
    def train_ratio(self) -> float:
        """Calculate training ratio.

        Returns:
            Fraction of dataflow for training.
        """
        return 1.0 - self.test_ratio - self.val_ratio

    def get_split_ratios(self) -> tuple[float, float, float]:
        """Get all split ratios.

        Returns:
            Tuple of (train, val, test) ratios.
        """
        return (self.train_ratio, self.val_ratio, self.test_ratio)
