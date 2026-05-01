"""State pattern objects for live optimization execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.optim

from .triggers import ITransitionTrigger


@dataclass(kw_only=True, slots=True)
class ActiveStage:
    """Live execution state for a single optimizer stage.

    Holds the running optimizer (which may be a ``ConcurrentOptimizer`` for
    concurrent parameter groups), optional scheduler, and the trigger that
    controls when this stage yields to the next one.

    Attributes:
        optimizer: The active optimizer. May be a ``ConcurrentOptimizer`` wrapping
            multiple sub-optimizers on disjoint parameter sets.
        scheduler: Optional learning rate scheduler (lr_scheduler or None).
        trigger: Transition trigger that signals advancing to the next stage.
        stage_index: Zero-indexed position in the overall program.
        name: Optional label for logging and debugging.
        scheduler_monitor: Lightning metric name to route to scheduler.step() (e.g. "val_loss").
            Only meaningful when scheduler is not None.
        scheduler_frequency: Lightning step frequency for the scheduler. Defaults to 1.
    """

    optimizer: torch.optim.Optimizer
    scheduler: object | None
    trigger: ITransitionTrigger
    stage_index: int
    name: str = ""
    scheduler_monitor: str = "val_loss"
    scheduler_frequency: int = 1


@dataclass(kw_only=True, slots=True)
class RunningOptimizerPolicy:
    """Top-level mutable state for the active optimization program.

    Tracks which stage is currently active and provides the interface
    for the controller to advance through the program.

    Attributes:
        stages: Tuple of ActiveStage objects (each may hold a ConcurrentOptimizer).
        active_index: Zero-based index of the currently active stage.
    """

    stages: tuple[ActiveStage, ...]
    active_index: int = 0

    @property
    def current(self) -> ActiveStage:
        """Return the currently active stage.

        Returns:
            The ActiveStage at active_index.
        """
        return self.stages[self.active_index]

    def advance(self) -> bool:
        """Advance to the next stage.

        Returns:
            True if advanced to the next stage, False if already at final stage.
        """
        if self.active_index < len(self.stages) - 1:
            self.active_index += 1
            return True
        return False

    @property
    def is_at_final_stage(self) -> bool:
        """Check if the program is at the final stage.

        Returns:
            True if active_index is at or beyond the last stage.
        """
        return self.active_index >= len(self.stages) - 1
