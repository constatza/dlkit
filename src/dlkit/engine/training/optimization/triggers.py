"""Stage transition triggers for optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal


class ITransitionTrigger(ABC):
    """Abstract interface for stage transition triggers.

    A trigger monitors training progress (by epoch and metrics) and signals
    when an optimization stage should transition to the next one. The caller
    is responsible for resetting the trigger after handling a transition.
    """

    @abstractmethod
    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Update trigger state and check if transition should fire.

        Args:
            epoch: The current training epoch number (0-indexed).
            metrics: Dict of current metric values (e.g., {"val_loss": 0.42}).

        Returns:
            True if the transition should fire now; False otherwise.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the trigger to its initial state.

        Called by the controller after a transition fires, allowing the trigger
        to be reused for the next stage transition.
        """
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return the trigger's state as a serializable dict.

        Used for checkpointing and restoring the trigger during training resumption.

        Returns:
            Dict containing all state needed to reconstruct the trigger.
        """
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the trigger's state from a checkpoint.

        Args:
            state: State dict previously returned by state_dict().
        """
        ...


class EpochTransitionTrigger(ITransitionTrigger):
    """Trigger that fires at a specific epoch.

    Fires exactly once when the current epoch reaches the target epoch.
    Subsequent calls return False until reset.

    Attributes:
        _at_epoch: Target epoch for transition.
        _fired: Whether the transition has already fired.
    """

    def __init__(self, at_epoch: int) -> None:
        """Initialize the epoch trigger.

        Args:
            at_epoch: The epoch number at which to fire (0-indexed).
        """
        self._at_epoch = at_epoch
        self._fired = False

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Check if we've reached the target epoch.

        Args:
            epoch: Current epoch number.
            metrics: Current metrics (unused).

        Returns:
            True if epoch >= self._at_epoch and not yet fired; False otherwise.
        """
        if not self._fired and epoch >= self._at_epoch:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        """Reset the fired flag for the next stage."""
        self._fired = False

    def state_dict(self) -> dict[str, Any]:
        """Return the trigger state.

        Returns:
            Dict with 'at_epoch' and 'fired' keys.
        """
        return {"at_epoch": self._at_epoch, "fired": self._fired}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trigger state.

        Args:
            state: State dict from state_dict().
        """
        self._at_epoch = state["at_epoch"]
        self._fired = state["fired"]


class PlateauTransitionTrigger(ITransitionTrigger):
    """Trigger that fires when a monitored metric plateaus.

    Monitors a single metric and counts consecutive epochs without improvement.
    Fires when the patience counter reaches the threshold.

    Attributes:
        _monitor: Name of the metric to monitor.
        _patience: Number of epochs to wait for improvement.
        _min_delta: Minimum change to qualify as improvement.
        _mode: "min" to minimize metric, "max" to maximize.
        _best: Best metric value seen so far.
        _counter: Current patience counter.
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        min_delta: float = 1e-4,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        """Initialize the plateau trigger.

        Args:
            monitor: Name of the metric to monitor (e.g., "val_loss").
            patience: Number of epochs without improvement before firing.
            min_delta: Minimum absolute change to count as improvement.
            mode: "min" if lower metric is better, "max" if higher is better.
        """
        self._monitor = monitor
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best: float | None = None
        self._counter = 0

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Update with current metric and check for plateau.

        Args:
            epoch: Current epoch (unused; uses metric value instead).
            metrics: Dict of metrics; must contain self._monitor key.

        Returns:
            True if patience counter has reached threshold; False otherwise.
        """
        current = metrics.get(self._monitor)
        if current is None:
            return False

        # Initialize best on first update
        if self._best is None:
            self._best = current
            return False

        # Check for improvement
        improved = self._is_improvement(current, self._best)

        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1

        return self._counter >= self._patience

    def reset(self) -> None:
        """Reset the trigger for the next stage."""
        self._best = None
        self._counter = 0

    def state_dict(self) -> dict[str, Any]:
        """Return the trigger state.

        Returns:
            Dict with monitor, patience, min_delta, mode, best, and counter.
        """
        return {
            "monitor": self._monitor,
            "patience": self._patience,
            "min_delta": self._min_delta,
            "mode": self._mode,
            "best": self._best,
            "counter": self._counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore trigger state.

        Args:
            state: State dict from state_dict().
        """
        self._monitor = state["monitor"]
        self._patience = state["patience"]
        self._min_delta = state["min_delta"]
        self._mode = state["mode"]
        self._best = state["best"]
        self._counter = state["counter"]

    def _is_improvement(self, current: float, best: float) -> bool:
        """Check if current metric represents improvement over best.

        Args:
            current: Current metric value.
            best: Best metric value so far.

        Returns:
            True if improvement threshold is met.
        """
        if self._mode == "min":
            return (best - current) > self._min_delta
        else:  # max
            return (current - best) > self._min_delta


class NoTransitionTrigger(ITransitionTrigger):
    """Trigger that never fires.

    Useful as a sentinel or placeholder when a stage should not transition.

    This is a no-op trigger — all methods either return False or do nothing.
    """

    def update(self, epoch: int, metrics: dict[str, float]) -> bool:
        """Never transition.

        Args:
            epoch: Current epoch (unused).
            metrics: Current metrics (unused).

        Returns:
            Always False.
        """
        return False

    def reset(self) -> None:
        """No-op."""
        pass

    def state_dict(self) -> dict[str, Any]:
        """Return empty state.

        Returns:
            Empty dict.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """No-op.

        Args:
            state: Ignored.
        """
        pass
