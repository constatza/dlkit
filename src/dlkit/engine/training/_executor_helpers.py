"""Helper functions for vanilla training execution."""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlkit.infrastructure.config.job_config import JobConfig

if TYPE_CHECKING:
    from lightning.pytorch.callbacks import ModelCheckpoint


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


def _checkpoint_artifacts_from_callback(callback: ModelCheckpoint) -> dict[str, Path]:
    """Resolve best/last checkpoint paths tracked by a ModelCheckpoint callback.

    Resolution precedence: `best_model_path`/`last_model_path` attributes first,
    falling back to globbing the callback's `dirpath` for `last.ckpt`/`*-last.ckpt`,
    then any other `*.ckpt` file.

    Args:
        callback: Lightning `ModelCheckpoint` callback instance.

    Returns:
        Dict with `"best_checkpoint"`/`"last_checkpoint"` keys for whichever
        paths could be resolved; missing entries are simply omitted.
    """
    artifacts: dict[str, Path] = {}

    if getattr(callback, "best_model_path", None):
        artifacts["best_checkpoint"] = Path(callback.best_model_path)

    if getattr(callback, "last_model_path", None):
        artifacts["last_checkpoint"] = Path(callback.last_model_path)

    if "last_checkpoint" in artifacts:
        return artifacts

    if not callback.dirpath:
        return artifacts

    dirpath = Path(callback.dirpath)
    if not dirpath.exists():
        return artifacts

    last_checkpoints = list(dirpath.glob("last.ckpt")) + list(dirpath.glob("*-last.ckpt"))
    if last_checkpoints:
        artifacts["last_checkpoint"] = last_checkpoints[0]
        return artifacts

    if "best_checkpoint" in artifacts:
        return artifacts

    ckpt_files = [f for f in dirpath.glob("*.ckpt") if "last" not in f.name]
    if ckpt_files:
        artifacts["best_checkpoint"] = ckpt_files[0]

    return artifacts


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
