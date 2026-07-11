"""Tests for the metric-key producer in step_logger.

Locks the contract between the metric-key producer (`_format_metric_name`)
and the canonical stage/key convention in `dlkit.common.metric_stages`, so a
future drift between the two (as previously happened with "val_loss" vs
"val/loss") is caught immediately.
"""

import pytest

from dlkit.common.metric_stages import MetricStage, metric_key
from dlkit.engine.adapters.lightning.concerns.step_logger import _format_metric_name
from dlkit.infrastructure.config.training_settings import StoppingSettings


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
