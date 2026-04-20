"""Tests for RunningOptimizerPolicy state object."""

from __future__ import annotations

from dlkit.engine.training.optimization.state import RunningOptimizerPolicy


class TestRunningOptimizerPolicyCurrent:
    """Tests for the current property."""

    def test_current_returns_stage_at_active_index(
        self, two_stage_program: RunningOptimizerPolicy
    ) -> None:
        """Verify current returns the stage at active_index.

        Args:
            two_stage_program: Two-stage fixture.
        """
        assert two_stage_program.current is two_stage_program.stages[0]
        two_stage_program.advance()
        assert two_stage_program.current is two_stage_program.stages[1]


class TestRunningOptimizerPolicyAdvance:
    """Tests for advancing through the program."""

    def test_advance_moves_to_next_stage(self, two_stage_program: RunningOptimizerPolicy) -> None:
        """Verify advance increments active_index and returns True.

        Args:
            two_stage_program: Two-stage fixture.
        """
        assert two_stage_program.active_index == 0
        result = two_stage_program.advance()
        assert result is True
        assert two_stage_program.active_index == 1

    def test_advance_returns_false_at_final_stage(
        self, two_stage_program: RunningOptimizerPolicy
    ) -> None:
        """Verify advance returns False when at final stage.

        Args:
            two_stage_program: Two-stage fixture.
        """
        two_stage_program.advance()  # Move to stage 1
        assert two_stage_program.active_index == 1
        result = two_stage_program.advance()
        assert result is False
        assert two_stage_program.active_index == 1


class TestRunningOptimizerPolicyFinalStage:
    """Tests for final stage detection."""

    def test_is_at_final_stage_false_at_start(
        self, two_stage_program: RunningOptimizerPolicy
    ) -> None:
        """Verify is_at_final_stage returns False at start.

        Args:
            two_stage_program: Two-stage fixture.
        """
        assert two_stage_program.is_at_final_stage is False

    def test_is_at_final_stage_true_after_advance(
        self, two_stage_program: RunningOptimizerPolicy
    ) -> None:
        """Verify is_at_final_stage returns True after advancing.

        Args:
            two_stage_program: Two-stage fixture.
        """
        two_stage_program.advance()
        assert two_stage_program.is_at_final_stage is True

    def test_single_stage_program_is_always_at_final(
        self, single_stage_program: RunningOptimizerPolicy
    ) -> None:
        """Verify single-stage program is always at final.

        Args:
            single_stage_program: Single-stage fixture.
        """
        assert single_stage_program.is_at_final_stage is True
        single_stage_program.advance()
        assert single_stage_program.is_at_final_stage is True
