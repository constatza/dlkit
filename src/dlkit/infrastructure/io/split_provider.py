"""Simple split index management with caching.

Provides a straightforward function to get or create splits with automatic caching.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from dlkit.infrastructure.io.index import load_split_indices, save_split_indices
from dlkit.infrastructure.io.locations import splits_dir
from dlkit.infrastructure.types.split import IndexSplit, Splitter


def _log_split_to_mlflow(split_file: Path) -> None:
    """Log a split file to the active MLflow run as a best-effort side effect.

    Only fires when ``mlflow.active_run()`` returns a live run. Silently
    swallows all exceptions so split generation is never blocked by tracking
    failures.

    Args:
        split_file: Path to the split JSON file to log.
    """
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        mlflow.log_artifact(str(split_file), artifact_path="splits")
        logger.debug(f"Logged split file to MLflow: {split_file}")
    except Exception as exc:
        logger.warning(f"Could not log split to MLflow (non-fatal): {exc}")


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

    # Strategy 2: Try loading cached split with size-aware filename
    split_file = splits_dir() / f"{session_name}_{num_samples}_split.json"
    if split_file.exists():
        try:
            logger.info(f"Loading cached split from {split_file}")
            cached_split = load_split_indices(split_file)

            # Validate cached split matches dataset size
            total_cached = (
                len(cached_split.train) + len(cached_split.validation) + len(cached_split.test)
            )
            if total_cached != num_samples:
                logger.warning(
                    f"Cached split size ({total_cached}) doesn't match dataset ({num_samples}). "
                    "Regenerating split."
                )
            else:
                return cached_split
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

    # Save for future use with size in filename
    try:
        save_split_indices(index_split, split_file)
        logger.info(f"Saved split indices to {split_file}")
        _log_split_to_mlflow(split_file)
    except Exception as e:
        logger.warning(f"Failed to save split to {split_file}: {e}")

    return index_split
