"""Shared workflow result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .result_accessors import NestedKey, TrainingResultAccessor

if TYPE_CHECKING:
    from dlkit.common.state import ModelState


@dataclass(frozen=True, slots=True, kw_only=True)
class TrialRecord:
    """Serializable record of a completed optimization trial.

    This is a value object that stores trial results for API compatibility.
    """

    number: int
    value: float | None
    params: dict[str, Any]
    state: str
    duration_seconds: float | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingResult:
    """Result of a training workflow execution."""

    model_state: ModelState | None
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    duration_seconds: float
    predictions: list[Any] | None = field(default=None)
    mlflow_run_id: str | None = field(default=None)
    mlflow_tracking_uri: str | None = field(default=None)
    _accessor: TrainingResultAccessor | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def checkpoint_path(self) -> Path | None:
        """Get the best or last checkpoint path from artifacts."""
        if "best_checkpoint" in self.artifacts:
            return self.artifacts["best_checkpoint"]
        if "last_checkpoint" in self.artifacts:
            return self.artifacts["last_checkpoint"]
        return None

    def _get_accessor(self) -> TrainingResultAccessor:
        accessor = self._accessor
        if accessor is None:
            accessor = TrainingResultAccessor(self)
            object.__setattr__(self, "_accessor", accessor)
        return accessor

    @property
    def stacked(self):
        """Lazily stacked prediction outputs."""
        return self._get_accessor().stacked

    def to_numpy(self, *keys: NestedKey):
        """Convert stacked prediction outputs to numpy arrays."""
        return self._get_accessor().to_numpy(*keys)


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceResult:
    """Result of an inference workflow execution."""

    model_state: ModelState | None
    predictions: Any
    metrics: dict[str, Any] | None
    duration_seconds: float


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationResult:
    """Result of hyperparameter optimization workflow."""

    best_trial: TrialRecord | None
    training_result: TrainingResult
    study_summary: dict[str, Any]
    duration_seconds: float
