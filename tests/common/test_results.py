"""OptimizationResult.as_training_result() exposes the wrapped TrainingResult explicitly."""

from __future__ import annotations

import pytest

from dlkit.common.errors import WorkflowError
from dlkit.common.results import OptimizationResult, TrainingResult


def test_as_training_result_returns_wrapped_training_result(
    optimization_result_with_training: OptimizationResult, training_result: TrainingResult
) -> None:
    assert optimization_result_with_training.as_training_result() is training_result


def test_own_duration_seconds_is_not_shadowed_by_training_result(
    optimization_result_with_training: OptimizationResult, training_result: TrainingResult
) -> None:
    """OptimizationResult's own duration_seconds (total search time) wins over
    the wrapped TrainingResult's (just the retrain's duration)."""
    assert optimization_result_with_training.duration_seconds == 300.0
    assert training_result.duration_seconds == 12.5


def test_as_training_result_raises_when_no_training_result(
    optimization_result_without_training: OptimizationResult,
) -> None:
    with pytest.raises(WorkflowError):
        optimization_result_without_training.as_training_result()
