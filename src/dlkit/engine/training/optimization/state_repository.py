"""Repository for checkpoint state round-trip of optimization programs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, cast

from .state import ActiveConcurrentGroup, ActiveStage, RunningOptimizationProgram


class IOptimizationStateRepository(ABC):
    """Abstract interface for persisting and restoring optimization state.

    Implementations handle the checkpoint/restore cycle for optimizer state,
    scheduler state, trigger state, and program progression.
    """

    @abstractmethod
    def save(self, program: RunningOptimizationProgram) -> dict[str, object]:
        """Serialize optimization program state to a checkpoint dict.

        Args:
            program: The running optimization program to serialize.

        Returns:
            Dict containing active_index, stage states (optimizer, scheduler, trigger).
        """
        ...

    @abstractmethod
    def restore(self, program: RunningOptimizationProgram, state: dict[str, object]) -> None:
        """Restore optimization program state from a checkpoint dict.

        Args:
            program: The running program to restore into (modified in-place).
            state: State dict previously returned by save().
        """
        ...


def _flatten_stages(program: RunningOptimizationProgram) -> tuple[ActiveStage, ...]:
    """Extract all ActiveStage objects from a program, flattening concurrent groups.

    Args:
        program: The program to flatten.

    Returns:
        Tuple of all ActiveStage objects in the program.
    """
    stages: list[ActiveStage] = []
    for entry in program.stages:
        if isinstance(entry, ActiveStage):
            stages.append(entry)
        elif isinstance(entry, ActiveConcurrentGroup):
            stages.extend(entry.stages)
    return tuple(stages)


class OptimizationStateRepository(IOptimizationStateRepository):
    """Concrete repository for optimization state checkpoint round-trip.

    Serializes optimizer state dicts, scheduler state dicts, trigger state,
    and the active stage index. Restores all components during checkpoint load.
    """

    def save(self, program: RunningOptimizationProgram) -> dict[str, object]:
        """Serialize the running program state.

        Args:
            program: The program to serialize.

        Returns:
            Dict with 'active_index' and 'stages' (list of stage state dicts).
        """
        all_stages = _flatten_stages(program)

        stages_state: list[dict[str, object]] = []
        for stage in all_stages:
            scheduler_state: dict[str, Any] | None = None
            if stage.scheduler is not None and hasattr(stage.scheduler, "state_dict"):
                scheduler_method = getattr(stage.scheduler, "state_dict", None)
                if callable(scheduler_method):
                    scheduler_state = cast(dict[str, Any], scheduler_method())

            stage_dict: dict[str, object] = {
                "optimizer_state": stage.optimizer.state_dict(),
                "scheduler_state": scheduler_state,
                "trigger_state": stage.trigger.state_dict(),
            }
            stages_state.append(stage_dict)

        return {
            "active_index": program.active_index,
            "stages": stages_state,
        }

    def restore(self, program: RunningOptimizationProgram, state: dict[str, object]) -> None:
        """Restore the program state from a checkpoint dict.

        Args:
            program: The program to restore into (modified in-place).
            state: State dict from save().
        """
        all_stages = _flatten_stages(program)
        stages_state_obj = state.get("stages", [])
        stages_state: list[Any] = (
            cast(list[Any], stages_state_obj) if isinstance(stages_state_obj, list) else []
        )

        # Restore each stage's optimizer, scheduler, and trigger
        for i, stage in enumerate(all_stages):
            if i < len(stages_state):
                stage_dict_obj = stages_state[i]
                stage_dict: dict[str, Any] = (
                    cast(dict[str, Any], stage_dict_obj) if isinstance(stage_dict_obj, dict) else {}
                )

                # Restore optimizer state
                if "optimizer_state" in stage_dict:
                    optimizer_state = cast(dict[str, Any], stage_dict["optimizer_state"])
                    if optimizer_state is not None and isinstance(optimizer_state, dict):
                        stage.optimizer.load_state_dict(optimizer_state)

                # Restore scheduler state if present
                if (
                    stage.scheduler is not None
                    and hasattr(stage.scheduler, "load_state_dict")
                    and "scheduler_state" in stage_dict
                    and stage_dict["scheduler_state"] is not None
                ):
                    scheduler_state = cast(dict[str, Any], stage_dict["scheduler_state"])
                    if isinstance(scheduler_state, dict):
                        load_method = getattr(stage.scheduler, "load_state_dict", None)
                        if callable(load_method):
                            load_method(scheduler_state)

                # Restore trigger state
                if "trigger_state" in stage_dict:
                    trigger_state = cast(dict[str, Any], stage_dict["trigger_state"])
                    if isinstance(trigger_state, dict):
                        stage.trigger.load_state_dict(trigger_state)

        # Restore active_index
        if "active_index" in state:
            active_idx = state["active_index"]
            if isinstance(active_idx, int):
                program.active_index = active_idx
