"""Functional extension points for MLflow tracking lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class TrackingHooks:
    """Functional extension points for MLflow tracking lifecycle.

    All callables are pure functions (no side effects on dlkit internals).
    Pass closures to inject behaviour without subclassing.

    Args:
        on_run_created: Called immediately after MLflow run opens.
            Receives (run_id, tracking_uri). Use to persist run metadata.
        on_training_complete: Called after core training finishes.
            Receives the (pre-enrichment) TrainingResult.
        extra_tags: Pure function settings -> dict[str, str].
            Tags are merged with tags from MLflowSettings (hooks win on collision).
        extra_params: Pure function settings -> dict[str, Any].
            Logged as MLflow params alongside model hyperparams.
        extra_artifacts: Pure function result -> Sequence[Path].
            Paths logged as artifacts after training.
    """

    on_run_created: Callable[[str, str | None], None] | None = field(default=None)
    on_training_complete: Callable[[Any], None] | None = field(default=None)
    extra_tags: Callable[[Any], dict[str, str]] | None = field(default=None)
    extra_params: Callable[[Any], dict[str, Any]] | None = field(default=None)
    extra_artifacts: Callable[[Any], Sequence[Path]] | None = field(default=None)
