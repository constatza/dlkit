"""Step policies for driving optimizers forward during training."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast, runtime_checkable

from torch import Tensor
from torch.optim import LBFGS

from .concurrent_optimizer import ConcurrentOptimizer
from .manual_host import IManualOptimizationHost, IManualOptimizer
from .state import ActiveStage


@runtime_checkable
class IStepPolicy(Protocol):
    """Protocol for optimizer stepping strategies."""

    def step(
        self,
        stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
        host: IManualOptimizationHost | None = None,
    ) -> Tensor:
        """Execute one optimizer step.

        Args:
            stage: The active stage to step.
            loss_fn: Callable that computes and returns the loss tensor.
            host: Optional Lightning-aware manual optimization host.

        Returns:
            The computed loss tensor.
        """
        ...


def _raw_step(stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
    """Execute a raw PyTorch optimizer step."""
    stage.optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    stage.optimizer.step()
    return loss


def _raw_lbfgs_step(stage: ActiveStage, loss_fn: Callable[[], Tensor]) -> Tensor:
    """Execute a raw PyTorch LBFGS step with a closure."""

    def closure() -> Tensor:
        stage.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    optimizer = cast(LBFGS, stage.optimizer)
    return cast(Tensor, optimizer.step(closure))


def _normalize_optimizers(optimizers: object | list[object]) -> list[IManualOptimizer]:
    """Normalize Lightning host optimizer output to a list."""
    if isinstance(optimizers, list):
        return [cast(IManualOptimizer, optimizer) for optimizer in optimizers]
    return [cast(IManualOptimizer, optimizers)]


def _resolve_host_optimizer(
    host: IManualOptimizationHost | None,
    stage_index: int,
) -> IManualOptimizer | None:
    """Resolve the active Lightning optimizer wrapper for the given stage."""
    if host is None:
        return None

    try:
        optimizers = host.optimizers(use_pl_optimizer=True)
    except RuntimeError:
        return None

    wrappers = _normalize_optimizers(optimizers)
    if stage_index >= len(wrappers):
        raise RuntimeError(
            f"Active stage index {stage_index} is out of range for {len(wrappers)} optimizer(s)."
        )
    return wrappers[stage_index]


def _manual_step(
    host: IManualOptimizationHost,
    optimizer: IManualOptimizer,
    loss_fn: Callable[[], Tensor],
) -> Tensor:
    """Execute one Lightning manual-optimization step."""
    with optimizer.toggle_model(sync_grad=True):
        optimizer.zero_grad()
        loss = loss_fn()
        host.manual_backward(loss)
        optimizer.step()
    return loss


def _manual_lbfgs_step(
    host: IManualOptimizationHost,
    optimizer: IManualOptimizer,
    loss_fn: Callable[[], Tensor],
) -> Tensor:
    """Execute one Lightning-aware LBFGS step with a closure."""

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = loss_fn()
        host.manual_backward(loss)
        return loss

    with optimizer.toggle_model(sync_grad=True):
        return cast(Tensor, optimizer.step(closure=closure))


def _stage_uses_lbfgs(stage: ActiveStage) -> bool:
    """Return True when the active stage requires closure-based LBFGS stepping."""
    if isinstance(stage.optimizer, LBFGS):
        return True
    if isinstance(stage.optimizer, ConcurrentOptimizer):
        return any(isinstance(optimizer, LBFGS) for optimizer in stage.optimizer.sub_optimizers)
    return False


class StepAllOptimizers:
    """Stepping policy that handles zero_grad, loss, backward, and step.

    Works with both plain and ConcurrentOptimizer stages — the optimizer
    interface is uniform, so no type dispatch is needed.
    """

    def step(
        self,
        stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
        host: IManualOptimizationHost | None = None,
    ) -> Tensor:
        """Step the stage optimizer.

        Args:
            stage: The active stage.
            loss_fn: Callable that computes the loss.
            host: Optional Lightning-aware manual optimization host.

        Returns:
            The computed loss tensor.
        """
        optimizer = _resolve_host_optimizer(host, stage.stage_index)
        if optimizer is None:
            return _raw_step(stage, loss_fn)
        if host is None:
            raise RuntimeError("host must not be None when optimizer is resolved")
        return _manual_step(host, optimizer, loss_fn)


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

    def step(
        self,
        stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
        host: IManualOptimizationHost | None = None,
    ) -> Tensor:
        """Step the stage optimizer.

        Args:
            stage: The active stage.
            loss_fn: Callable that computes the loss.
            host: Optional Lightning-aware manual optimization host.

        Returns:
            The computed loss tensor.
        """
        optimizer = _resolve_host_optimizer(host, stage.stage_index)
        if optimizer is None:
            loss = _raw_step(stage, loss_fn)
        else:
            assert host is not None
            loss = _manual_step(host, optimizer, loss_fn)
        self._step_counter += 1
        return loss


class LBFGSStageStepper:
    """Stepping policy for LBFGS optimizer (requires a closure)."""

    def step(
        self,
        stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
        host: IManualOptimizationHost | None = None,
    ) -> Tensor:
        """Step using LBFGS closure contract.

        Args:
            stage: The active stage with an LBFGS optimizer.
            loss_fn: Callable that computes the loss.
            host: Optional Lightning-aware manual optimization host.

        Returns:
            The computed loss tensor.
        """
        if not _stage_uses_lbfgs(stage):
            optimizer = _resolve_host_optimizer(host, stage.stage_index)
            if optimizer is None:
                return _raw_step(stage, loss_fn)
            if host is None:
                raise RuntimeError("host must not be None when optimizer is resolved")
            return _manual_step(host, optimizer, loss_fn)

        optimizer = _resolve_host_optimizer(host, stage.stage_index)
        if optimizer is None:
            return _raw_lbfgs_step(stage, loss_fn)
        if host is None:
            raise RuntimeError("host must not be None when optimizer is resolved")
        return _manual_lbfgs_step(host, optimizer, loss_fn)
