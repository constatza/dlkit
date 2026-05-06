"""Protocols for Lightning-aware manual optimization helpers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal, Protocol, overload, runtime_checkable

from torch import Tensor


@runtime_checkable
class IManualOptimizer(Protocol):
    """Optimizer wrapper contract used by manual optimization step policies."""

    param_groups: list[dict[str, Any]]

    def zero_grad(self) -> None:
        """Clear accumulated gradients."""

    def step(self, closure: Callable[[], Any] | None = None, **kwargs: Any) -> Any:
        """Advance the optimizer by one step."""

    def toggle_model(self, sync_grad: bool = True) -> AbstractContextManager[None]:
        """Return a context manager that toggles the active parameter set."""


@runtime_checkable
class IManualOptimizationHost(Protocol):
    """Minimal host capabilities needed for Lightning manual optimization."""

    def manual_backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        """Backpropagate through the manual-optimization backend."""

    @overload
    def optimizers(
        self, use_pl_optimizer: Literal[True] = True
    ) -> IManualOptimizer | list[IManualOptimizer]: ...

    @overload
    def optimizers(self, use_pl_optimizer: Literal[False]) -> object | list[object]: ...

    def optimizers(self, use_pl_optimizer: bool = True) -> object | list[object]:
        """Return configured optimizers for the current training run."""
