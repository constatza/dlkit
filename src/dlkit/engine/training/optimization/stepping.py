"""Step policies for driving optimizers forward during training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from torch import Tensor

from .state import ActiveStage


@runtime_checkable
class IStepPolicy(Protocol):
    """Protocol for optimizer stepping strategies."""

    def step(self, stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Execute one optimizer step.

        Args:
            stage: The active stage to step.
            loss_fn: Callable that computes and returns the loss tensor.

        Returns:
            The computed loss tensor.
        """
        ...


class StepAllOptimizers:
    """Stepping policy that handles zero_grad, loss, backward, and step.

    Works with both plain and ConcurrentOptimizer stages — the optimizer
    interface is uniform, so no type dispatch is needed.
    """

    def step(self, stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Step the stage optimizer.

        Args:
            stage: The active stage.
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """
        stage.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        stage.optimizer.step()
        return loss


class AlternatingStepPolicy:
    """Alternating stepping policy — cycles through sequential stages.

    Period-based rotation: steps stage at index
    ``(step_counter // period) % len(stages)`` for multi-stage programs.
    For single-stage use, behaves like StepAllOptimizers.

    Attributes:
        _period: Number of calls before advancing to the next stage.
        _step_counter: Tracks total calls across all steps.
    """

    def __init__(self, period: int = 1) -> None:
        """Initialize the alternating policy.

        Args:
            period: Steps per stage before rotating.
        """
        self._period = period
        self._step_counter = 0

    def step(self, stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Step the stage optimizer.

        Args:
            stage: The active stage.
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """
        stage.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        stage.optimizer.step()
        self._step_counter += 1
        return loss


class LBFGSStageStepper:
    """Stepping policy for LBFGS optimizer (requires a closure)."""

    def step(self, stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
        """Step using LBFGS closure contract.

        Args:
            stage: The active stage with an LBFGS optimizer.
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """

        def closure() -> Tensor:
            stage.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        return stage.optimizer.step(closure)  # type: ignore[return-value]
