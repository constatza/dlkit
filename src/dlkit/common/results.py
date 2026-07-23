"""Shared workflow result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .errors import WorkflowError
from .result_accessors import NestedKey, TrainingResultAccessor

if TYPE_CHECKING:
    from dlkit.common.state import ModelState

# How a multirun sweep should react when one child run fails: stop immediately,
# keep going and report failures alongside successes, or keep going but mark
# the parent run as failed once all children have finished.
FailurePolicy = Literal["fail_fast", "continue", "continue_mark_parent_failed"]


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
class EvaluationResult:
    """Result of evaluating a trained checkpoint against a labeled dataset split.

    ``predictions``/``targets``/``figures`` are ``Any``-typed (numpy arrays /
    matplotlib Figures) to keep numpy and matplotlib out of ``common``'s
    import surface, matching ``TrainingResult.predictions``.
    """

    predictions: Any
    targets: Any
    metrics: dict[str, float]
    figures: dict[str, Any]
    duration_seconds: float
    mlflow_run_id: str | None = field(default=None)
    mlflow_tracking_uri: str | None = field(default=None)


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiRunResult[T]:
    """Parent run id plus one record per child run.

    Generic so both eval's batch results and (potentially, as a later,
    separate follow-up) MultiRunOrchestrator.run_sweep()'s currently-bare
    tuple[TrainingResult, ...] can share one container shape instead of
    each inventing its own.

    Args:
        parent_run_id: MLflow run id of the parent sweep run.
        children: One record per child run, in execution order.
        tracking_uri: MLflow tracking URI the parent and children were
            recorded under, if tracked. Defaults to ``None`` for callers
            that predate this field.
    """

    parent_run_id: str
    children: tuple[T, ...]
    tracking_uri: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildFailure:
    """Serializable failure record for one child of a multirun sweep.

    Deliberately holds no raw exception object (which may be unpicklable or
    reference process-local state) so it can be logged, returned across
    process boundaries, and pattern-matched via ``status`` alongside
    :class:`ChildSuccess` as part of :data:`ChildOutcome`.

    Args:
        child_id: Identifier of the child run within the sweep.
        exception_type: Name of the exception type that caused the failure.
        message: Human-readable failure message.
        run_id: MLflow run id of the failed child, if one was created.
        stage: Workflow stage the failure occurred in, if known.
        status: Discriminant literal, always ``"failure"``.
    """

    child_id: str
    exception_type: str
    message: str
    run_id: str | None
    stage: str | None
    status: Literal["failure"] = "failure"


@dataclass(frozen=True, slots=True, kw_only=True)
class ChildSuccess[T]:
    """Successful outcome for one child of a multirun sweep.

    Args:
        child_id: Identifier of the child run within the sweep.
        run_id: MLflow run id of the successful child, if one was created.
        result: The child's workflow result.
        status: Discriminant literal, always ``"success"``.
    """

    child_id: str
    run_id: str | None
    result: T
    status: Literal["success"] = "success"


type ChildOutcome[T] = ChildSuccess[T] | ChildFailure


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationResult:
    """Result of hyperparameter optimization workflow."""

    best_trial: TrialRecord | None
    training_result: TrainingResult | None
    study_summary: dict[str, Any]
    duration_seconds: float
    mlflow_run_id: str | None = field(default=None)
    mlflow_tracking_uri: str | None = field(default=None)

    def as_training_result(self) -> TrainingResult:
        """Return the best trial's training result.

        Raises:
            WorkflowError: If no successful trial produced a training result.
        """
        if self.training_result is None:
            raise WorkflowError("No successful trial produced a training result to retrain")
        return self.training_result


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvergencePoint:
    """Stats for a single sample-size step in a convergence study.

    Args:
        n: Number of training samples for this step.
        train_mean: Mean training metric across repeats.
        train_std: Standard deviation of training metric; 0.0 when repeats=1.
        val_mean: Mean validation metric across repeats.
        val_std: Standard deviation of validation metric; 0.0 when repeats=1.
        gap: Generalisation gap (val_mean - train_mean).
        marginal_gain: Improvement in val_mean vs previous step; None for first point.
        converged: Whether convergence threshold is met; None if no target is set.
    """

    n: int
    train_mean: float
    train_std: float
    val_mean: float
    val_std: float
    gap: float
    marginal_gain: float | None
    converged: bool | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ConvergenceResult:
    """Result of a sample-size convergence workflow.

    Args:
        points: Ordered sequence of convergence points, one per evaluated size.
        n_star: Smallest n where converged=True; None if not found or no target set.
        duration_seconds: Total wall-clock time for the convergence study.
        mlflow_run_id: MLflow run ID for the parent convergence run, if tracked.
        mlflow_tracking_uri: MLflow tracking URI used during the run, if tracked.
    """

    points: tuple[ConvergencePoint, ...]
    n_star: int | None
    duration_seconds: float
    mlflow_run_id: str | None = field(default=None)
    mlflow_tracking_uri: str | None = field(default=None)


# Closed union of DLKit's per-workflow result types, used as multirun's generic
# child result type since a single sweep can mix workflow kinds.
WorkflowResult = TrainingResult | OptimizationResult | ConvergenceResult
