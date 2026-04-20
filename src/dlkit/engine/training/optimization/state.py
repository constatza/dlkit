"""State pattern objects for live optimization execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.optim

from .triggers import ITransitionTrigger


@dataclass(kw_only=True)
class ActiveStage:
    """Live execution state for a single optimizer stage.

    Holds the running optimizer, optional scheduler, and the trigger
    that controls when this stage yields to the next one.

    Attributes:
        optimizer: The active torch.optim.Optimizer instance.
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


@dataclass(kw_only=True)
class ActiveConcurrentGroup:
    """Live execution state for a group of concurrent optimizers.

    All optimizers in the group are stepped on every training step,
    operating on disjoint parameter sets.

    Attributes:
        stages: Tuple of ActiveStage objects running concurrently.
        trigger: Transition trigger that signals advancing beyond this group.
        group_index: Zero-indexed position in the overall program.
    """

    stages: tuple[ActiveStage, ...]
    trigger: ITransitionTrigger
    group_index: int


@dataclass(kw_only=True)
class RunningOptimizerPolicy:
    """Top-level mutable state for the active optimization program.

    Tracks which stage is currently active and provides the interface
    for the controller to advance through the program.

    Attributes:
        stages: Tuple of ActiveStage or ActiveConcurrentGroup objects.
        active_index: Zero-based index of the currently active stage/group.
    """

    stages: tuple[ActiveStage | ActiveConcurrentGroup, ...]
    active_index: int = 0

    @property
    def current(self) -> ActiveStage | ActiveConcurrentGroup:
        """Return the currently active stage or concurrent group.

        Returns:
            The ActiveStage or ActiveConcurrentGroup at active_index.
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
