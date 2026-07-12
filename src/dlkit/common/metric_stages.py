"""Canonical stage names and metric-key convention for train/val/test logging.

Single source of truth for which metric-key prefixes correspond to which
Lightning stage, and for how stage-prefixed metric keys are built. Used by
the metric producer (engine.adapters.lightning.concerns.step_logger), the
Lightning epoch-logging callback (engine.adapters.lightning.callbacks.MLflowEpochLogger),
the tracking-side metric summary logger (engine.tracking.metric_logger), and
config defaults that reference a metric name (StoppingSettings.monitor,
SearchSettings.objective, ConvergenceSettings.target_metric) — lives in
`common` because those layers cannot import from each other in the direction
this sharing would otherwise require.

Metric keys use "/" as the stage separator (e.g. "val/loss"), matching
MLflow's native hierarchical metric grouping (MLflow 2.11+ groups metrics by
"/"-delimited prefix in its UI, the same convention MLflow uses internally
for its own "system/*" metrics) — not "_", which produces a flat, ungrouped
metric list in MLflow's run view.
"""

from __future__ import annotations

from enum import StrEnum


class MetricStage(StrEnum):
    """Canonical Lightning stage identifiers used as metric-key prefixes."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


STAGE_ALIASES: dict[MetricStage, tuple[str, ...]] = {
    MetricStage.TRAIN: ("train", "training"),
    MetricStage.VAL: ("val", "valid", "validation"),
    MetricStage.TEST: ("test", "testing"),
}

METRIC_KEY_SEPARATOR = "/"


def metric_key(stage: MetricStage, name: str) -> str:
    """Build a canonical, MLflow-grouping-friendly metric key: '<stage>/<name>'.

    Args:
        stage: Canonical stage identifier (e.g. MetricStage.VAL). Only a
            `MetricStage` is accepted — a raw string here previously let
            already-malformed stage identifiers (e.g. "val_epoch") flow
            straight through into MLflow metric keys undetected.
        name: Raw metric name (e.g. "loss").

    Returns:
        Stage-prefixed metric key, e.g. "val/loss".
    """
    return f"{stage}{METRIC_KEY_SEPARATOR}{name}"


def swap_stage(key: str, from_stage: MetricStage, to_stage: MetricStage) -> str:
    """Swap a metric key's stage prefix, e.g. 'val/loss' -> 'train/loss'.

    Args:
        key: Metric key to transform.
        from_stage: Stage prefix expected in `key`.
        to_stage: Stage prefix to substitute in its place.

    Returns:
        `key` with its stage prefix swapped, or `key` unchanged if
        `from_stage`'s prefix isn't present.
    """
    return key.replace(metric_key(from_stage, ""), metric_key(to_stage, ""))


DEFAULT_VAL_LOSS_METRIC = metric_key(MetricStage.VAL, "loss")
DEFAULT_TRAIN_LOSS_METRIC = metric_key(MetricStage.TRAIN, "loss")
