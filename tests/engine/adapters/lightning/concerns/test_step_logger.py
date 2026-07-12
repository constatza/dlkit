"""Tests for the metric-key producer in step_logger.

Locks the contract between the metric-key producer (`_format_metric_name`)
and the canonical stage/key convention in `dlkit.common.metric_stages`, so a
future drift between the two (as previously happened with "val_loss" vs
"val/loss") is caught immediately.
"""

from typing import Any

import pytest

from dlkit.common.metric_stages import MetricStage, metric_key
from dlkit.engine.adapters.lightning.concerns.step_logger import (
    LightningStepLogger,
    _format_metric_name,
)
from dlkit.infrastructure.config.training_settings import StoppingSettings


class RecordingModule:
    """Fake LightningModule capturing log()/log_dict() calls for assertions."""

    def __init__(self) -> None:
        self.trainer: object = object()
        self.logged: dict[str, Any] = {}
        self.logged_dicts: dict[str, Any] = {}

    def log(self, name: str, value: Any, **kwargs: Any) -> None:
        self.logged[name] = value

    def log_dict(self, metrics: dict[str, Any], **kwargs: Any) -> None:
        self.logged_dicts.update(metrics)


@pytest.fixture
def recording_module() -> RecordingModule:
    return RecordingModule()


@pytest.fixture
def step_logger(recording_module: RecordingModule) -> LightningStepLogger:
    return LightningStepLogger(recording_module)


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        (MetricStage.TRAIN, "loss"),
        (MetricStage.VAL, "loss"),
        (MetricStage.TEST, "loss"),
    ],
)
def test_format_metric_name_prefixes_with_slash(stage: MetricStage, name: str) -> None:
    """A raw metric name is prefixed with its stage using the '/' separator."""
    assert _format_metric_name(stage, name) == metric_key(stage, name)


@pytest.mark.parametrize(
    ("stage", "name"),
    [
        (MetricStage.VAL, "val_something"),
        (MetricStage.VAL, "validation_accuracy"),
        (MetricStage.TRAIN, "training_loss"),
        (MetricStage.TEST, "testing_accuracy"),
    ],
)
def test_format_metric_name_does_not_double_prefix(stage: MetricStage, name: str) -> None:
    """A name already starting with one of the stage's aliases is returned unchanged."""
    assert _format_metric_name(stage, name) == name


def test_stopping_settings_monitor_matches_step_logger_output() -> None:
    """StoppingSettings.monitor must match the key step_logger actually produces.

    This is the assertion that directly encodes the contract broken by the
    original bug: the early-stopping config default silently referenced a
    metric key ("val_loss") that the step logger never produced ("val/loss").
    """
    expected_key = _format_metric_name(MetricStage.VAL, "loss")
    assert StoppingSettings().monitor == expected_key
    assert StoppingSettings().monitor == metric_key(MetricStage.VAL, "loss")


def test_log_lr_logs_under_train_group(
    step_logger: LightningStepLogger, recording_module: RecordingModule
) -> None:
    """Learning rate is a training-loop property, so it belongs under train/."""
    step_logger.log_lr(0.01)
    assert recording_module.logged[metric_key(MetricStage.TRAIN, "lr")] == 0.01


def test_log_walltime_logs_under_walltime_group_not_stage_group(
    step_logger: LightningStepLogger, recording_module: RecordingModule
) -> None:
    """Timing is kept out of the stage's own metric-performance group.

    "walltime/train" (group first), not "train/walltime" or "train/epoch_time_s"
    — timing must not mix into the same MLflow UI group as train/loss, train/lr.
    """
    step_logger.log_walltime(MetricStage.TRAIN, 1.5)
    assert recording_module.logged[f"walltime/{MetricStage.TRAIN}"] == 1.5


def test_log_stage_outputs_epoch_metrics_not_suffixed(
    step_logger: LightningStepLogger, recording_module: RecordingModule
) -> None:
    """Regression test: epoch-computed metrics must use the plain stage prefix.

    Previously ``_on_eval_epoch_end`` passed "val_epoch"/"test_epoch" as the
    stage, producing keys like "val_epoch/Accuracy" instead of the canonical
    "val/Accuracy" that monitor/objective defaults actually look up.
    """
    step_logger.log_stage_outputs(MetricStage.VAL, None, {"Accuracy": 0.9})
    assert recording_module.logged_dicts == {metric_key(MetricStage.VAL, "Accuracy"): 0.9}
