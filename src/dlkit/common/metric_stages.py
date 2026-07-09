"""Canonical stage-name aliases for training/validation metric key matching.

Single source of truth for which metric-key prefixes correspond to which
Lightning stage. Used by both the Lightning epoch-logging callback
(engine.adapters.lightning.callbacks.MLflowEpochLogger) and the
tracking-side metric summary logger (engine.tracking.metric_logger) —
lives in `common` because those two layers cannot import from each other
in the direction this sharing would otherwise require.
"""

from __future__ import annotations

STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
}
