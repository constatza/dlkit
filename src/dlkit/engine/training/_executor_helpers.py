"""Helper functions for vanilla training execution."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

from dlkit.infrastructure.config.job_config import JobConfig


def _get_seed(settings: JobConfig) -> int:
    """Extract seed from JobConfig."""
    return settings.run.seed or 42


def _get_lr_tuner(settings: JobConfig) -> Any | None:
    """Extract LR tuner settings from JobConfig."""
    return settings.training.lr_tuner if settings.training else None


def _get_optimizer(settings: JobConfig) -> Any | None:
    """Extract optimizer settings from JobConfig."""
    return settings.training.optimizer if settings.training else None


def _get_trainer_settings(settings: JobConfig) -> Any | None:
    """Extract trainer settings from JobConfig."""
    return settings.training.trainer if settings.training else None


@contextmanager
def _suppress_training_runtime_warnings():
    """Suppress known framework warnings that add noise during successful runs."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weights_only.*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="The '.*_dataloader' does not have many workers.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*isinstance\(treespec, LeafSpec\).*is deprecated.*",
        )
        yield
