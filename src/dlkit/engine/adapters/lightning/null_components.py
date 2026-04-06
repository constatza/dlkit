"""Null object implementations for Lightning wrapper components.

Used by wrappers that override all step methods and never delegate
to the base class invoker/computer/updater logic.
"""

from __future__ import annotations

from typing import Any

from tensordict import TensorDict
from torch import Tensor, nn


class _NullModelInvoker:
    """No-op model invoker for wrappers that override all step methods.

    Used by wrappers (e.g., GraphLightningWrapper) that pass their own
    batch format to the model and never delegate to the base step methods.
    """

    def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
        """Raise if accidentally called.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "_NullModelInvoker.invoke() was called unexpectedly. "
            "Subclass must override all step methods when using null invoker."
        )


class _NullLossComputer:
    """No-op loss computer for wrappers that override all step methods."""

    def compute(self, predictions: Tensor, batch: Any) -> Tensor:
        """Raise if accidentally called.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "_NullLossComputer.compute() was called unexpectedly. "
            "Subclass must override all step methods when using null computer."
        )


class _NullMetricsUpdater:
    """No-op metrics updater for wrappers that handle metrics directly."""

    def update(self, predictions: Tensor, batch: Any, stage: str) -> None:
        """No-op update."""

    def compute(self, stage: str) -> dict[str, Any]:
        """Return empty metrics dict.

        Args:
            stage: Stage identifier.

        Returns:
            Empty dict.
        """
        return {}

    def reset(self, stage: str) -> None:
        """No-op reset."""
