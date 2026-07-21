"""Split index resolution: seeded producer vs. strictly-reloading consumer.

Two functions, deliberately incompatible signatures (Interface Segregation):
``resolve_training_split`` may generate a fresh split and always has the
parameters (``num_samples``/ratios/``seed``) needed to do so;
``resolve_evaluation_split`` has none of those parameters and therefore
cannot generate a fresh split even by accident — it can only reload one that
already exists on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlkit.infrastructure.io.index import load_split_indices, save_split_indices
from dlkit.infrastructure.types.split import (
    IndexSplit,
    RatioSplitStrategy,
    SplitStrategy,
)
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExternalFileSplitStrategy:
    """Loads a pre-computed IndexSplit from a JSON or TOML file.

    Lives in the ``io`` layer (not alongside ``RatioSplitStrategy`` in
    ``infrastructure.types.split``) because it depends on
    ``load_split_indices``, and ``io`` already depends on ``types`` for
    ``IndexSplit``/``SplitStrategy`` — colocating this here keeps the
    ``types -> io`` edge from existing at all, avoiding a cycle.
    """

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


def _cap_train_indices(
    resolution: SplitResolution,
    *,
    max_train_samples: int,
    train_subset_seed: int | None,
) -> SplitResolution:
    """Truncate a resolved split's train indices to ``max_train_samples``.

    Args:
        resolution: Split resolution whose train indices will be capped.
        max_train_samples: Maximum number of training samples to keep.
        train_subset_seed: Seed for re-permuting train indices before
            capping. When None, the original index order is preserved.

    Returns:
        A new SplitResolution with the train split capped; val/test/predict
        and source metadata are unchanged.
    """
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
    return SplitResolution(
        index_split=capped_split,
        source_path=resolution.source_path,
        artifact_filename=resolution.artifact_filename,
    )


def resolve_training_split(
    *,
    num_samples: int,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    persist_to: Path | None,
    session_name: str = "default",
    explicit_filepath: Path | None = None,
    max_train_samples: int | None = None,
    train_subset_seed: int | None = None,
) -> SplitResolution:
    """Resolve (and optionally persist) the split used to train a run.

    Producer only: builds a fresh, seeded split unless an explicit split
    file is given. Always writes the resolved split to ``persist_to`` (when
    not None) so there is exactly one canonical place evaluation can later
    reload it from — this is the fix for the split-reproducibility bug where
    evaluation silently regenerated an unrelated random split.

    Args:
        num_samples: Total number of samples in the dataset.
        test_ratio: Fraction of the dataset used for testing.
        val_ratio: Fraction of the dataset used for validation.
        seed: Seed for the local RNG used to permute indices when generating
            a fresh split. Ignored when ``explicit_filepath`` is given.
        persist_to: Local path to write the resolved split to via
            ``save_split_indices``. Skipped entirely when None.
        session_name: Session identifier for split file naming.
        explicit_filepath: Optional path to an existing split file. When
            given, the split is loaded from this file instead of generated.
        max_train_samples: Optional cap on the number of training samples.
            When set, train indices are truncated to this size after
            optional re-permutation. Useful for convergence studies
            requiring nested training subsets.
        train_subset_seed: Seed for re-permuting train indices before
            capping. When None, the original index order is preserved.

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
        logger.info(
            "Generating new split for session '{}' ({} samples, seed={})",
            session_name,
            num_samples,
            seed,
        )
        strategy = RatioSplitStrategy(
            num_samples=num_samples,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed,
        )
        resolution = SplitResolution(
            index_split=strategy.split(),
            source_path=None,
            artifact_filename=f"{session_name}_{num_samples}_split.json",
        )

    if max_train_samples is not None:
        resolution = _cap_train_indices(
            resolution,
            max_train_samples=max_train_samples,
            train_subset_seed=train_subset_seed,
        )

    if persist_to is not None:
        logger.info("Persisting resolved split to {}", persist_to)
        save_split_indices(resolution.index_split, persist_to)

    return resolution


def resolve_evaluation_split(*, filepath: Path) -> SplitResolution:
    """Reload the exact split used to train a run, for evaluation.

    Consumer only: this function structurally cannot generate a fresh
    split — no ``num_samples``/ratio/``seed`` parameters exist on it at all,
    so an evaluation call site can never accidentally fall back to
    regenerating an unrelated random split.

    Args:
        filepath: Path to the persisted split file (JSON or TOML).

    Returns:
        SplitResolution loaded verbatim from ``filepath``.

    Raises:
        Exception: Whatever ``load_split_indices`` raises for a missing or
            malformed split file (propagated unchanged, fail loud).
    """
    strategy = ExternalFileSplitStrategy(filepath)
    return SplitResolution(
        index_split=strategy.split(),
        source_path=filepath,
        artifact_filename=filepath.name,
    )
