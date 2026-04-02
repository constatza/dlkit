"""Utilities for metrics collection and conversion."""

from __future__ import annotations

from typing import Any


def collect_metrics(source: dict[str, Any] | None) -> dict[str, Any]:
    """Convert metric dictionary values to plain Python types.

    This utility safely converts metric values (often tensors or special numeric types)
    to plain Python floats where possible, falling back to the original value if
    conversion fails.

    Args:
        source: Dictionary of metrics to collect. Can be None.

    Returns:
        Dictionary with converted metric values. Returns empty dict if source is None.

    Example:
        >>> import torch
        >>> metrics = {"loss": torch.tensor(0.5), "accuracy": 0.95}
        >>> collect_metrics(metrics)
        {'loss': 0.5, 'accuracy': 0.95}
    """
    collected: dict[str, Any] = {}
    if not source:
        return collected

    for key, value in source.items():
        try:
            collected[key] = float(value)
        except Exception:
            # Keep original value if conversion fails
            collected[key] = value

    return collected
