"""Simple split index management with caching.

Provides a straightforward function to get or create splits with automatic caching.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from dlkit.core.datatypes.split import IndexSplit, Splitter
from dlkit.tools.io.index import load_split_indices, save_split_indices
from dlkit.tools.io.locations import splits_dir


def get_or_create_split(
    *,
    num_samples: int,
    test_ratio: float,
    val_ratio: float,
    session_name: str = "default",
    explicit_filepath: Path | None = None,
) -> IndexSplit:
    """Get or create index split with automatic caching.

    Priority order:
    1. Explicit file path (if provided)
    2. Cached split file for this session
    3. Generate new split and cache it

    Args:
        num_samples: Total number of samples in dataset
        test_ratio: Fraction for test set
        val_ratio: Fraction for validation set
        session_name: Session identifier for split file naming
        explicit_filepath: Optional path to specific split file

    Returns:
        IndexSplit with train/val/test indices
    """
    # Strategy 1: Use explicit file if provided
    if explicit_filepath is not None:
        logger.info(f"Loading split indices from {explicit_filepath}")
        return load_split_indices(explicit_filepath)

    # Strategy 2: Try loading cached split
    split_file = splits_dir() / f"{session_name}_split.json"
    if split_file.exists():
        try:
            logger.info(f"Loading cached split from {split_file}")
            return load_split_indices(split_file)
        except Exception as e:
            logger.warning(f"Failed to load split from {split_file}: {e}. Generating new split.")

    # Strategy 3: Generate new split and cache
    logger.info(f"Generating new split for session '{session_name}' ({num_samples} samples)")
    splitter = Splitter(
        num_samples=num_samples,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
    )
    index_split = splitter.split()

    # Save for future use
    try:
        save_split_indices(index_split, split_file)
        logger.info(f"Saved split indices to {split_file}")
    except Exception as e:
        logger.warning(f"Failed to save split to {split_file}: {e}")

    return index_split
