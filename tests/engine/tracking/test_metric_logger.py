"""Tests for MetricLogger.log_summary_metrics.

Verifies that train/val metrics (already epoch-logged by MLflowEpochLogger)
are excluded here, while test metrics — a single scalar over the whole test
set, not an epoch-indexed series — pass through and are logged exactly once
with no step.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from dlkit.common import TrainingResult
from dlkit.common.hooks import ParamValue
from dlkit.engine.tracking.metric_logger import MetricLogger


class FakeRunContext:
    """Minimal IMetricSink double capturing logged metrics/params."""

    def __init__(self) -> None:
        self.logged_metrics: dict[str, float] = {}
        self.logged_params: dict[str, ParamValue] = {}
        self.log_metrics_calls = 0

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        assert step is None
        self.log_metrics_calls += 1
        self.logged_metrics.update(metrics)

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self.logged_params.update(dict(params))


@pytest.fixture
def metric_logger() -> MetricLogger:
    return MetricLogger()


@pytest.fixture
def run_context() -> FakeRunContext:
    return FakeRunContext()


@pytest.fixture
def mixed_stage_metrics() -> dict[str, float]:
    return {
        "train_loss": 0.9,
        "val/loss": 0.5,
        "validation_accuracy": 0.8,
        "test_loss": 0.3,
        "test/accuracy": 0.95,
        "status test": 1.0,
    }


@pytest.fixture
def training_result(mixed_stage_metrics: dict[str, float]) -> TrainingResult:
    return TrainingResult(
        model_state=None,
        metrics=mixed_stage_metrics,
        artifacts={},
        duration_seconds=0.0,
    )


def test_train_val_metrics_excluded(
    metric_logger: MetricLogger,
    training_result: TrainingResult,
    run_context: FakeRunContext,
) -> None:
    metric_logger.log_summary_metrics(training_result, run_context)

    assert "train_loss" not in run_context.logged_metrics
    assert "val/loss" not in run_context.logged_metrics
    assert "validation_accuracy" not in run_context.logged_metrics


def test_test_metrics_pass_through_once_with_no_step(
    metric_logger: MetricLogger,
    training_result: TrainingResult,
    run_context: FakeRunContext,
) -> None:
    metric_logger.log_summary_metrics(training_result, run_context)

    assert run_context.logged_metrics == {
        "test_loss": pytest.approx(0.3),
        "test/accuracy": pytest.approx(0.95),
        "status test": pytest.approx(1.0),
    }
    assert run_context.log_metrics_calls == 1


def test_no_metrics_logs_nothing(metric_logger: MetricLogger, run_context: FakeRunContext) -> None:
    result = TrainingResult(model_state=None, metrics={}, artifacts={}, duration_seconds=0.0)

    metric_logger.log_summary_metrics(result, run_context)

    assert run_context.logged_metrics == {}
    assert run_context.log_metrics_calls == 0
