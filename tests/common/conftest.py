"""Shared fixtures for dlkit.common value-object tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from dlkit.common.results import OptimizationResult, TrainingResult


@pytest.fixture
def training_result() -> TrainingResult:
    return TrainingResult(
        model_state=None,
        metrics={"val/loss": 0.1},
        artifacts={"best_checkpoint": Path("model.ckpt")},
        duration_seconds=12.5,
    )


@pytest.fixture
def optimization_result_with_training(training_result: TrainingResult) -> OptimizationResult:
    return OptimizationResult(
        best_trial=None,
        training_result=training_result,
        study_summary={"n_trials": 5},
        duration_seconds=300.0,
    )


@pytest.fixture
def optimization_result_without_training() -> OptimizationResult:
    return OptimizationResult(
        best_trial=None,
        training_result=None,
        study_summary={"n_trials": 5},
        duration_seconds=300.0,
    )
