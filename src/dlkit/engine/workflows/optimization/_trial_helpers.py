"""Pure helper utilities for optimization trial state transitions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from dlkit.common import TrainingResult
from dlkit.engine.workflows.optimization.value_objects import Trial, TrialState


def complete_trial(
    trial: Trial,
    *,
    hyperparameters: dict[str, object],
    objective_value: float,
    training_result: TrainingResult,
) -> Trial:
    """Return a completed trial record."""
    return replace(
        trial,
        hyperparameters=hyperparameters,
        objective_value=objective_value,
        training_result=training_result,
        state=TrialState.COMPLETE,
        completed_at=datetime.now(),
    )


def prune_trial(
    trial: Trial,
    *,
    hyperparameters: dict[str, object],
    pruned_at_step: int,
) -> Trial:
    """Return a pruned trial record."""
    return replace(
        trial,
        hyperparameters=hyperparameters,
        state=TrialState.PRUNED,
        pruned_at_step=pruned_at_step,
        completed_at=datetime.now(),
    )


def fail_trial(
    trial: Trial,
    *,
    hyperparameters: dict[str, object],
) -> Trial:
    """Return a failed trial record."""
    return replace(
        trial,
        hyperparameters=hyperparameters,
        state=TrialState.FAILED,
        completed_at=datetime.now(),
    )
