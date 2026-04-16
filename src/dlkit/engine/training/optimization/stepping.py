"""Step policies for driving optimizers forward during training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from torch import Tensor

from .state import ActiveConcurrentGroup, ActiveStage


@runtime_checkable
class IStepPolicy(Protocol):
    """Protocol for optimizer stepping strategies.

    Implementations define how to compute loss, backward, and step optimizers.
    """

    def step(
        self, stage: ActiveStage | ActiveConcurrentGroup, loss_fn: Callable[[], Tensor]
    ) -> Tensor:
        """Execute one optimizer step(s).

        Args:
            stage: The active stage or concurrent group to step.
            loss_fn: Callable that computes and returns the loss tensor.

        Returns:
            The computed loss tensor.
        """
        ...


class StepAllOptimizers:
    """Simple stepping policy that steps all optimizers in a group.

    For concurrent groups, steps all stages in order.
    For single stages, steps the single optimizer.

    Handles zero_grad, loss computation, backward, and optimizer.step().
    """

    def step(
        self, stage: ActiveStage | ActiveConcurrentGroup, loss_fn: Callable[[], Tensor]
    ) -> Tensor:
        """Step all optimizers in the stage or group.

        Args:
            stage: The active stage or concurrent group.
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """
        if isinstance(stage, ActiveConcurrentGroup):
            # Step all optimizers in the group
            for sub_stage in stage.stages:
                sub_stage.optimizer.zero_grad()

            loss = loss_fn()
            loss.backward()

            for sub_stage in stage.stages:
                sub_stage.optimizer.step()

            return loss

        else:  # ActiveStage
            stage.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            stage.optimizer.step()
            return loss


class AlternatingStepPolicy:
    """Alternating stepping policy for concurrent groups.

    Cycles through stages in a concurrent group with a configurable period.
    Each call steps the stage at index (step_counter // period) % len(stages).

    For single stages, behaves like StepAllOptimizers.

    Attributes:
        _period: Number of calls before advancing to the next stage.
        _step_counter: Tracks total calls across all steps.
    """

    def __init__(self, period: int = 1) -> None:
        """Initialize the alternating policy.

        Args:
            period: Steps per stage before rotating (period=1 steps every stage each call).
        """
        self._period = period
        self._step_counter = 0

    def step(
        self, stage: ActiveStage | ActiveConcurrentGroup, loss_fn: Callable[[], Tensor]
    ) -> Tensor:
        """Step one stage, rotating through concurrent group.

        Args:
            stage: The active stage or concurrent group.
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.
        """
        if isinstance(stage, ActiveConcurrentGroup):
            # Determine which stage to step
            stage_index = (self._step_counter // self._period) % len(stage.stages)
            active_stage = stage.stages[stage_index]

            active_stage.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            active_stage.optimizer.step()

            self._step_counter += 1
            return loss

        else:  # ActiveStage
            stage.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            stage.optimizer.step()
            self._step_counter += 1
            return loss


class LBFGSStageStepper:
    """Stepping policy for LBFGS optimizer.

    LBFGS requires a closure function rather than separate backward/step.
    This policy wraps the loss function in the required closure contract.
    """

    def step(
        self, stage: ActiveStage | ActiveConcurrentGroup, loss_fn: Callable[[], Tensor]
    ) -> Tensor:
        """Step using LBFGS closure contract.

        Args:
            stage: The active stage (must be single optimizer, not concurrent group).
            loss_fn: Callable that computes the loss.

        Returns:
            The computed loss tensor.

        Raises:
            TypeError: If stage is a concurrent group (LBFGS requires single optimizer).
        """
        if isinstance(stage, ActiveConcurrentGroup):
            raise TypeError("LBFGSStageStepper does not support concurrent groups")

        def closure() -> Tensor:
            stage.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        # LBFGS.step() expects a closure and returns the loss
        return stage.optimizer.step(closure)  # type: ignore[return-value]
