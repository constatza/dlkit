"""Split index management without implicit local persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)

from dlkit.infrastructure.io.index import load_split_indices
from dlkit.infrastructure.types.split import IndexSplit, RatioSplitStrategy, SplitStrategy


@dataclass(frozen=True, slots=True)
class ExternalFileSplitStrategy:
    """Loads a pre-computed IndexSplit from a JSON or TOML file."""

    filepath: Path

    def split(self) -> IndexSplit:
        return load_split_indices(self.filepath)


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
    max_train_samples: int | None = None,
    train_subset_seed: int | None = None,
) -> SplitResolution:
    """Get an index split, using an explicit file when provided.

    Generated splits remain in memory by default. Local persistence is opt-in
    via ``explicit_filepath`` only.

    Args:
        num_samples: Total number of samples in dataset.
        test_ratio: Fraction for test set.
        val_ratio: Fraction for validation set.
        session_name: Session identifier for split file naming.
        explicit_filepath: Optional path to specific split file.
        max_train_samples: Optional cap on the number of training samples. When set,
            train indices are truncated to this size after optional re-permutation.
            Useful for convergence studies requiring nested training subsets.
        train_subset_seed: Seed for re-permuting train indices before capping.
            When None, the original index order is preserved.

    Returns:
        SplitResolution containing the split and optional source file metadata.
    """
    if explicit_filepath is not None:
        logger.info("Loading split indices from {}", explicit_filepath)
        strategy: SplitStrategy = ExternalFileSplitStrategy(explicit_filepath)
        resolution = SplitResolution(
            index_split=strategy.split(),
            source_path=explicit_filepath,
            artifact_filename=explicit_filepath.name,
        )
    else:
        logger.info("Generating new split for session '{}' ({} samples)", session_name, num_samples)
        strategy = RatioSplitStrategy(
            num_samples=num_samples,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
        )
        resolution = SplitResolution(
            index_split=strategy.split(),
            source_path=None,
            artifact_filename=f"{session_name}_{num_samples}_split.json",
        )

    if max_train_samples is not None:
        import numpy as np

        train_indices = list(resolution.index_split.train)
        if train_subset_seed is not None:
            rng = np.random.default_rng(train_subset_seed)
            train_indices = rng.permutation(train_indices).tolist()
        capped_split = IndexSplit(
            train=tuple(train_indices[:max_train_samples]),
            validation=resolution.index_split.validation,
            test=resolution.index_split.test,
            predict=resolution.index_split.predict,
        )
        resolution = SplitResolution(
            index_split=capped_split,
            source_path=resolution.source_path,
            artifact_filename=resolution.artifact_filename,
        )

    return resolution
