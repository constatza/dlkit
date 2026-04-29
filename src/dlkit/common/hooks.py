"""Shared workflow hook contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .results import TrainingResult


@dataclass(frozen=True, slots=True, kw_only=True)
class LifecycleHooks:
    """Functional extension points for lifecycle events."""

    on_run_created: Callable[[str, str | None], None] | None = field(default=None)
    on_training_complete: Callable[[TrainingResult], None] | None = field(default=None)
    extra_tags: Callable[[TrainingResult], dict[str, str]] | None = field(default=None)
    extra_params: Callable[[TrainingResult], dict[str, object]] | None = field(default=None)
    extra_artifacts: Callable[[TrainingResult], Sequence[Path]] | None = field(default=None)
