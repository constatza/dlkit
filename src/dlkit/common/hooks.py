"""Shared workflow hook contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .results import TrainingResult

# Extensible scalar parameter value for runtime metadata surfaces.
# This is a sum type, not a renaming alias.
type ParamValue = str | int | float | bool

# Which MLflow run a RunCreatedEvent describes. "study"/"train"/"sweep"/"evaluate"
# are the outermost run of a dlkit call when not itself nested inside another
# active run; "trial"/"best_retrain" are always nested children of a study run.
RunKind = Literal["study", "trial", "best_retrain", "train", "sweep", "evaluate"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RunCreatedEvent:
    """Fired via ``LifecycleHooks.on_run_created`` when a new MLflow run opens.

    ``is_outermost`` is a real field, not derived from ``kind`` alone: a
    ``"train"`` run's outermost-ness depends on whether an MLflow run was
    already active when it was created, which isn't knowable from ``kind``
    by itself.
    """

    run_id: str
    tracking_uri: str | None
    kind: RunKind
    is_outermost: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class LifecycleHooks:
    """Functional extension points for lifecycle events."""

    on_run_created: Callable[[RunCreatedEvent], None] | None = field(default=None)
    on_training_complete: Callable[[TrainingResult], None] | None = field(default=None)
    extra_tags: Callable[[TrainingResult], dict[str, str]] | None = field(default=None)
    extra_params: Callable[[TrainingResult], dict[str, ParamValue]] | None = field(default=None)
    extra_artifacts: Callable[[TrainingResult], Sequence[Path]] | None = field(default=None)
