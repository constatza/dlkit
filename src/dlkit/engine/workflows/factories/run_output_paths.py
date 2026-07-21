"""Single source of truth for a run's local file-based artifact output root."""

from __future__ import annotations

from pathlib import Path

from dlkit.infrastructure.config.job_config import JobConfig


def resolve_local_artifact_root(settings: JobConfig) -> Path | None:
    """Resolve the local root directory for this run's file-based artifacts.

    Mirrors the ``training.trainer.default_root_dir`` convention already used
    to pin Lightning-owned local outputs (checkpoints, loggers) under one
    directory. Reused here as the single source of truth for where locally
    persisted split files live, so both concerns read from the same place
    instead of duplicating the lookup.

    A unique ``default_root_dir`` per run is required for reliable later
    recovery: reusing the same root directory across runs silently
    overwrites a previous run's local split/checkpoint files (the same
    collision risk that already exists for the fixed ``"best"`` checkpoint
    filename).

    Args:
        settings: Full job configuration.

    Returns:
        The configured ``default_root_dir`` as a ``Path``, or None when no
        training section, trainer section, or ``default_root_dir`` is
        configured.
    """
    training = settings.training
    if training is None:
        return None
    trainer = training.trainer
    if trainer is None:
        return None
    default_root_dir = trainer.default_root_dir
    if default_root_dir is None:
        return None
    return Path(default_root_dir)
