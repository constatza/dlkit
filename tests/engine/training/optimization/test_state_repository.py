"""Tests for OptimizationStateRepository checkpoint round-trip."""

from __future__ import annotations

from typing import Any, cast

import pytest

from dlkit.engine.training.optimization.concurrent_optimizer import ConcurrentOptimizer
from dlkit.engine.training.optimization.state import ActiveStage, RunningOptimizerPolicy
from dlkit.engine.training.optimization.state_repository import (
    OptimizationStateRepository,
)


@pytest.fixture
def repository() -> OptimizationStateRepository:
    """Provide a fresh OptimizationStateRepository for each test.

    Returns:
        A new OptimizationStateRepository instance.
    """
    return OptimizationStateRepository()


class TestOptimizationStateRepositorySave:
    """Tests for the save method."""

    def test_save_contains_active_index(
        self,
        single_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify saved state contains active_index.

        Args:
            single_stage_program: Single-stage fixture.
            repository: Repository fixture.
        """
        saved = repository.save(single_stage_program)
        assert "active_index" in saved
        assert saved["active_index"] == 0

    def test_save_contains_stages(
        self,
        single_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify saved state contains stages list.

        Args:
            single_stage_program: Single-stage fixture.
            repository: Repository fixture.
        """
        saved = repository.save(single_stage_program)
        assert "stages" in saved
        assert isinstance(saved["stages"], list)
        assert len(saved["stages"]) > 0

    def test_save_stage_has_optimizer_state(
        self,
        single_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify each stage state contains optimizer_state dict.

        Args:
            single_stage_program: Single-stage fixture.
            repository: Repository fixture.
        """
        saved = repository.save(single_stage_program)
        stages = cast(list[Any], saved["stages"])
        assert isinstance(stages, list)
        assert len(stages) > 0
        stage_dict = cast(dict[str, Any], stages[0])
        assert "optimizer_state" in stage_dict
        assert isinstance(stage_dict["optimizer_state"], dict)

    def test_save_stage_has_trigger_state(
        self,
        single_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify each stage state contains trigger_state dict.

        Args:
            single_stage_program: Single-stage fixture.
            repository: Repository fixture.
        """
        saved = repository.save(single_stage_program)
        stages = cast(list[Any], saved["stages"])
        assert isinstance(stages, list)
        assert len(stages) > 0
        stage_dict = cast(dict[str, Any], stages[0])
        assert "trigger_state" in stage_dict
        assert isinstance(stage_dict["trigger_state"], dict)


class TestOptimizationStateRepositoryRestore:
    """Tests for the restore method."""

    def test_restore_restores_active_index(
        self,
        two_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify restore sets active_index from checkpoint.

        Args:
            two_stage_program: Two-stage fixture.
            repository: Repository fixture.
        """
        # Save at initial state
        saved = repository.save(two_stage_program)
        assert saved["active_index"] == 0

        # Advance program
        two_stage_program.advance()
        assert two_stage_program.active_index == 1

        # Restore from saved state
        repository.restore(two_stage_program, saved)
        assert two_stage_program.active_index == 0

    def test_restore_restores_trigger_state(
        self,
        two_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify restore loads trigger state and resets _fired flag.

        Args:
            two_stage_program: Two-stage fixture.
            repository: Repository fixture.
        """
        # Save initial state (trigger not fired)
        saved = repository.save(two_stage_program)

        # Fire the trigger by calling update at target epoch
        stage = two_stage_program.stages[0]
        trigger = stage.trigger
        trigger.update(5, {})
        trigger_state_after_fire = trigger.state_dict()

        # Verify trigger was fired
        assert trigger_state_after_fire.get("fired") is True

        # Restore from saved state
        repository.restore(two_stage_program, saved)
        trigger_state_restored = trigger.state_dict()

        # Verify trigger was restored to pre-fire state
        assert trigger_state_restored.get("fired") is False

    def test_round_trip_preserves_optimizer_state(
        self,
        single_stage_program: RunningOptimizerPolicy,
        repository: OptimizationStateRepository,
    ) -> None:
        """Verify optimizer state is preserved through save/restore cycle.

        Args:
            single_stage_program: Single-stage fixture.
            repository: Repository fixture.
        """
        import torch

        stage = single_stage_program.stages[0]
        optimizer = stage.optimizer

        # Run an optimization step to populate optimizer state
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                param.grad = torch.ones_like(param)
        optimizer.step()
        optimizer.zero_grad()

        # Save state with populated optimizer
        saved = repository.save(single_stage_program)
        stages = cast(list[Any], saved["stages"])
        stage_dict = cast(dict[str, Any], stages[0])
        optimizer_state_saved = cast(dict[str, Any], stage_dict["optimizer_state"])

        # Verify saved state contains param groups and state
        assert "param_groups" in optimizer_state_saved
        assert isinstance(optimizer_state_saved["param_groups"], list)

        # Manually clear optimizer state
        optimizer.state.clear()

        # Verify state is now empty
        assert len(optimizer.state) == 0

        # Restore from saved state
        repository.restore(single_stage_program, saved)

        # Verify optimizer state is restored (has state entries)
        assert len(optimizer.state) > 0


# ---------------------------------------------------------------------------
# Task C: Concurrent group checkpoint / restore round-trip
# ---------------------------------------------------------------------------


class TestConcurrentGroupRoundTrip:
    """State repository round-trip for an ActiveStage with ConcurrentOptimizer."""

    def test_restore_preserves_all_sub_optimizer_learning_rates(
        self,
        concurrent_group: ActiveStage,
        repository: OptimizationStateRepository,
    ) -> None:
        """Save / restore must recover original LRs for every sub-optimizer.

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer containing two SGDs.
            repository: Fresh repository fixture.
        """
        assert isinstance(concurrent_group.optimizer, ConcurrentOptimizer)
        program = RunningOptimizerPolicy(stages=(concurrent_group,))

        sub_opts = concurrent_group.optimizer.sub_optimizers
        original_lrs = [opt.param_groups[0]["lr"] for opt in sub_opts]

        saved = repository.save(program)

        # Corrupt all LRs to a sentinel value
        for opt in sub_opts:
            for pg in opt.param_groups:
                pg["lr"] = 0.999

        repository.restore(program, saved)

        for i, opt in enumerate(sub_opts):
            assert opt.param_groups[0]["lr"] == pytest.approx(original_lrs[i])

    def test_restore_preserves_active_index(
        self,
        concurrent_group: ActiveStage,
        repository: OptimizationStateRepository,
    ) -> None:
        """Save / restore must recover the program's active_index correctly.

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer.
            repository: Fresh repository fixture.
        """
        program = RunningOptimizerPolicy(stages=(concurrent_group,), active_index=0)
        saved = repository.save(program)

        program.active_index = 99

        repository.restore(program, saved)
        assert program.active_index == 0

    def test_save_concurrent_stage_serializes_optimizer_state(
        self,
        concurrent_group: ActiveStage,
        repository: OptimizationStateRepository,
    ) -> None:
        """Save must serialize the ConcurrentOptimizer state (sub_optimizers key).

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer.
            repository: Fresh repository fixture.
        """
        assert isinstance(concurrent_group.optimizer, ConcurrentOptimizer)
        program = RunningOptimizerPolicy(stages=(concurrent_group,))
        saved = repository.save(program)

        stages_state = cast(list[Any], saved["stages"])
        assert len(stages_state) == 1
        stage_dict = cast(dict[str, Any], stages_state[0])
        assert "optimizer_state" in stage_dict
        assert "trigger_state" in stage_dict
        # ConcurrentOptimizer.state_dict() has "sub_optimizers" key
        opt_state = cast(dict[str, Any], stage_dict["optimizer_state"])
        assert "sub_optimizers" in opt_state
        assert len(opt_state["sub_optimizers"]) == 2  # noqa: PLR2004
