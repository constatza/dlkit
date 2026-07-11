"""Tests for the MLflowEpochLogger callback."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from lightning import Callback
from lightning.pytorch import LightningModule

from dlkit.common.hooks import ParamValue
from dlkit.engine.adapters.lightning.callbacks import MLflowEpochLogger


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
        self.params: dict[str, ParamValue] = {}
        self.tags: dict[str, str] = {}

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged.append((metrics, step))

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self.params.update(dict(params))

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


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


def test_epoch_logger_has_no_test_hook() -> None:
    """Test metrics are not epoch-indexed; MLflowEpochLogger must not log them.

    See MetricLogger.log_summary_metrics for the single, step-less test-metric
    logging path.
    """
    assert not hasattr(MLflowEpochLogger, "on_test_end") or (
        MLflowEpochLogger.on_test_end is Callback.on_test_end
    )


def test_train_only_metrics_collected_on_train_excluded_on_val(
    run_context: RecordingRunContext,
) -> None:
    """train/lr is train-scoped; excluded from val's push.

    Regression test for removing the hardcoded ``normalized_key == "lr"``
    special case in ``_collect_stage_metrics``: since the metric now carries
    a real "train/" prefix, plain prefix matching handles it with no
    special-casing needed.
    """
    logger = MLflowEpochLogger(run_context)
    pl_module = cast("LightningModule", Mock(spec=LightningModule))

    trainer = _build_trainer(
        epoch=1,
        metrics={
            "train/loss": FakeTensor(0.5),
            "train/lr": FakeTensor(0.001),
            "val/loss": FakeTensor(0.4),
        },
    )

    logger.on_train_epoch_end(trainer, pl_module=pl_module)
    train_metrics, _ = run_context.logged[-1]
    assert train_metrics == {
        "train/loss": pytest.approx(0.5),
        "train/lr": pytest.approx(0.001),
    }

    run_context.logged.clear()
    logger.on_validation_epoch_end(trainer, pl_module=pl_module)
    val_metrics, _ = run_context.logged[-1]
    assert val_metrics == {"val/loss": pytest.approx(0.4)}


def test_walltime_key_attributed_only_to_its_own_stage(
    run_context: RecordingRunContext,
) -> None:
    """Group-first keys like "walltime/train" are matched by trailing stage segment.

    "walltime/train" doesn't start with "train", so the plain-prefix check
    alone wouldn't attribute it to the train push or exclude it from val's —
    covers the ``endswith(f"/{prefix}")`` branch added to `_matches_stage_prefix`.
    """
    logger = MLflowEpochLogger(run_context)
    pl_module = cast("LightningModule", Mock(spec=LightningModule))

    trainer = _build_trainer(
        epoch=1,
        metrics={
            "train/loss": FakeTensor(0.5),
            "walltime/train": FakeTensor(12.3),
            "val/loss": FakeTensor(0.4),
        },
    )

    logger.on_train_epoch_end(trainer, pl_module=pl_module)
    train_metrics, _ = run_context.logged[-1]
    assert train_metrics == {
        "train/loss": pytest.approx(0.5),
        "walltime/train": pytest.approx(12.3),
    }

    run_context.logged.clear()
    logger.on_validation_epoch_end(trainer, pl_module=pl_module)
    val_metrics, _ = run_context.logged[-1]
    assert val_metrics == {"val/loss": pytest.approx(0.4)}


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
