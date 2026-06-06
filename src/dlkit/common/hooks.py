"""Shared workflow hook contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .results import TrainingResult

# Extensible scalar parameter value for runtime metadata surfaces.
# This is a sum type, not a renaming alias.
type ParamValue = str | int | float | bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LifecycleHooks:
    """Functional extension points for lifecycle events."""

    on_run_created: Callable[[str, str | None], None] | None = field(default=None)
    on_training_complete: Callable[[TrainingResult], None] | None = field(default=None)
    extra_tags: Callable[[TrainingResult], dict[str, str]] | None = field(default=None)
    extra_params: Callable[[TrainingResult], dict[str, ParamValue]] | None = field(default=None)
    extra_artifacts: Callable[[TrainingResult], Sequence[Path]] | None = field(default=None)
