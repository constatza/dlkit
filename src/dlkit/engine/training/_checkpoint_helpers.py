"""Checkpoint path resolution helper for training execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dlkit.infrastructure.utils.logging_config import get_logger

if TYPE_CHECKING:
    from dlkit.infrastructure.config.job_config import JobConfig

logger = get_logger(__name__)


def _resolve_checkpoint_path(settings: JobConfig) -> str | None:
    """Resolve a resume-from-checkpoint path from training settings.

    Combines raw config extraction with file-existence validation and
    appropriate logging. Returns None when no checkpoint is configured
    or the configured path does not exist on disk.

    Args:
        settings: A JobConfig instance.

    Returns:
        Absolute checkpoint path as a string when the file exists, None otherwise.
    """
    training = settings.training
    if training is None or not training.resume_from_checkpoint:
        return None

    checkpoint_path = Path(str(training.resume_from_checkpoint))
    if not checkpoint_path.exists():
        logger.warning(
            "Training checkpoint configured but not found: %s. Starting training from scratch.",
            checkpoint_path,
        )
        return None

    logger.info("Resuming training from checkpoint: %s", checkpoint_path)
    return str(checkpoint_path)
