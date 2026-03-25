"""Tests for the MLflowEpochLogger callback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from lightning.pytorch import LightningModule

from dlkit.core.training.callbacks import MLflowEpochLogger


class FakeTensor:
    """Tensor-like helper that mimics Lightning scalar outputs."""

    def __init__(self, value: float):
        self._value = value
        self.detached = False
        self.moved_to_cpu = False

    def detach(self) -> FakeTensor:
        self.detached = True
        return self

    def cpu(self) -> FakeTensor:
        self.moved_to_cpu = True
        return self

    def item(self) -> float:
        return self._value


class RecordingRunContext:
    """Minimal run context to capture metrics for assertions."""

    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged.append((metrics, step))


@pytest.fixture
def run_context() -> RecordingRunContext:
    return RecordingRunContext()


def _build_trainer(epoch: int, metrics: dict[str, FakeTensor], sanity_checking: bool = False):
    return SimpleNamespace(
        current_epoch=epoch,
        callback_metrics=metrics,
        sanity_checking=sanity_checking,
    )


def test_validation_metrics_logged_per_epoch(run_context: RecordingRunContext) -> None:
    logger = MLflowEpochLogger(run_context)
    pl_module = cast("LightningModule", Mock(spec=LightningModule))

    trainer = _build_trainer(
        epoch=2,
        metrics={
            "val/loss": FakeTensor(0.42),
            "validation_accuracy": FakeTensor(0.91),
            "Accuracy": FakeTensor(0.77),
            "train/loss": FakeTensor(0.33),
        },
    )

    logger.on_validation_epoch_end(trainer, pl_module=pl_module)

    assert len(run_context.logged) == 1
    metrics, step = run_context.logged[0]
    assert step == 2
    assert metrics == {
        "val/loss": pytest.approx(0.42),
        "validation_accuracy": pytest.approx(0.91),
        "Accuracy": pytest.approx(0.77),
    }
    assert trainer.callback_metrics["val/loss"].detached is True
    assert trainer.callback_metrics["val/loss"].moved_to_cpu is True


def test_test_metrics_logged_once(run_context: RecordingRunContext) -> None:
    logger = MLflowEpochLogger(run_context)
    pl_module = cast("LightningModule", Mock(spec=LightningModule))

    trainer = _build_trainer(
        epoch=5,
        metrics={
            "test/loss": FakeTensor(0.25),
            "test_accuracy": FakeTensor(0.88),
            "Accuracy": FakeTensor(0.79),
            "val/loss": FakeTensor(0.5),
        },
    )

    logger.on_test_end(trainer, pl_module=pl_module)

    assert len(run_context.logged) == 1
    metrics, step = run_context.logged[0]
    assert step == 5
    assert metrics == {
        "test/loss": pytest.approx(0.25),
        "test_accuracy": pytest.approx(0.88),
        "Accuracy test": pytest.approx(0.79),
    }
    assert trainer.callback_metrics["test/loss"].detached is True
    assert trainer.callback_metrics["test/loss"].moved_to_cpu is True


def test_sanity_checking_skips_logging(run_context: RecordingRunContext) -> None:
    logger = MLflowEpochLogger(run_context)
    pl_module = cast("LightningModule", Mock(spec=LightningModule))

    trainer = _build_trainer(
        epoch=0,
        metrics={"val/loss": FakeTensor(0.1)},
        sanity_checking=True,
    )

    logger.on_validation_epoch_end(trainer, pl_module=pl_module)

    assert run_context.logged == []
