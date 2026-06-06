"""Split index management without implicit local persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from dlkit.infrastructure.io.index import load_split_indices
from dlkit.infrastructure.types.split import IndexSplit, Splitter


@dataclass(frozen=True, slots=True, kw_only=True)
class SplitResolution:
    """Resolved split payload plus optional source artifact metadata."""

    index_split: IndexSplit
    source_path: Path | None
    artifact_filename: str

    @property
    def has_explicit_file(self) -> bool:
        return self.source_path is not None


def get_or_create_split(
    *,
    num_samples: int,
    test_ratio: float,
    val_ratio: float,
    session_name: str = "default",
    explicit_filepath: Path | None = None,
) -> SplitResolution:
    """Get an index split, using an explicit file when provided.

    Generated splits remain in memory by default. Local persistence is opt-in
    via ``explicit_filepath`` only.

    Args:
        num_samples: Total number of samples in dataset
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set
        session_name: Session identifier for split file naming
        explicit_filepath: Optional path to specific split file

    Returns:
        SplitResolution containing the split and optional source file metadata.
    """
    if explicit_filepath is not None:
        logger.info(f"Loading split indices from {explicit_filepath}")
        return SplitResolution(
            index_split=load_split_indices(explicit_filepath),
            source_path=explicit_filepath,
            artifact_filename=explicit_filepath.name,
        )

    logger.info(f"Generating new split for session '{session_name}' ({num_samples} samples)")
    splitter = Splitter(
        num_samples=num_samples,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )
    return SplitResolution(
        index_split=splitter.split(),
        source_path=None,
        artifact_filename=f"{session_name}_{num_samples}_split.json",
    )
