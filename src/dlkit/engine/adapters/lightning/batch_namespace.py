"""Batch namespace contracts for TensorDict-based Lightning wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class IBatchNamespaceSpec(Protocol):
    """Declares the TensorDict namespace keys used for features and targets.

    Inject a custom implementation to support non-standard batch layouts.
    The default is ``StandardBatchNamespace`` (``"features"`` / ``"targets"``).
    """

    @property
    def feature_namespace(self) -> str:
        """Namespace key for feature tensors in the batch TensorDict."""
        ...

    @property
    def target_namespace(self) -> str:
        """Namespace key for target tensors in the batch TensorDict."""
        ...


@dataclass(frozen=True)
class StandardBatchNamespace:
    """Default batch namespace using ``"features"`` and ``"targets"`` keys.

    Attributes:
        feature_namespace: Key for feature tensors (default ``"features"``).
        target_namespace: Key for target tensors (default ``"targets"``).
    """

    feature_namespace: str = "features"
    target_namespace: str = "targets"


def _parse_key(key: str) -> tuple[str, str]:
    """Parse 'namespace.entry_name' key into (namespace, entry_name).

    Args:
        key: Key string in 'features.name' or 'targets.name' format.

    Returns:
        Tuple of (namespace, entry_name).

    Raises:
        ValueError: If key format is invalid.
    """
    parts = key.split(".", 1)
    if len(parts) != 2 or parts[0] not in ("features", "targets"):
        raise ValueError(
            f"key must be 'features.<entry_name>' or 'targets.<entry_name>', got '{key}'"
        )
    return parts[0], parts[1]
