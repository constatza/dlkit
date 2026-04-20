"""Tests for optimization controllers (automatic and manual modes)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from dlkit.engine.training.optimization.controllers import (
    AutomaticOptimizationController,
    ManualOptimizationController,
)
from dlkit.engine.training.optimization.state import (
    RunningOptimizerPolicy,
)
from dlkit.engine.training.optimization.state_repository import OptimizationStateRepository
from dlkit.engine.training.optimization.stepping import StepAllOptimizers


# Local fixtures
@pytest.fixture
def repository() -> OptimizationStateRepository:
    """Create a fresh optimization state repository.

    Returns:
        An OptimizationStateRepository instance.
    """
    return OptimizationStateRepository()


@pytest.fixture
def auto_controller(
    single_stage_program: RunningOptimizerPolicy,
    repository: OptimizationStateRepository,
) -> AutomaticOptimizationController:
    """Create an automatic optimization controller.

    Args:
        single_stage_program: Single-stage optimization program fixture.
        repository: Repository for state persistence.

    Returns:
        An AutomaticOptimizationController instance.
    """
    return AutomaticOptimizationController(single_stage_program, repository)


@pytest.fixture
def manual_controller(
    single_stage_program: RunningOptimizerPolicy,
    repository: OptimizationStateRepository,
) -> ManualOptimizationController:
    """Create a manual optimization controller.

    Args:
        single_stage_program: Single-stage optimization program fixture.
        repository: Repository for state persistence.

    Returns:
        A ManualOptimizationController instance.
    """
    return ManualOptimizationController(single_stage_program, repository, StepAllOptimizers())


@pytest.fixture
def auto_controller_two_stages(
    two_stage_program: RunningOptimizerPolicy,
    repository: OptimizationStateRepository,
) -> AutomaticOptimizationController:
    """Create an automatic controller with two stages.

    Args:
        two_stage_program: Two-stage optimization program fixture.
        repository: Repository for state persistence.

    Returns:
        An AutomaticOptimizationController instance with two stages.
    """
    return AutomaticOptimizationController(two_stage_program, repository)


# Test AutomaticOptimizationController
class TestAutomaticOptimizationController:
    """Tests for AutomaticOptimizationController."""

    def test_auto_controller_requires_manual_optimization_is_false(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """Verify that automatic controller reports False for manual optimization.

        Args:
            auto_controller: Automatic controller fixture.
        """
        assert auto_controller.requires_manual_optimization is False

    def test_auto_configure_optimizers_single_stage_returns_dict(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """Verify single-stage configuration returns dict format.

        Args:
            auto_controller: Automatic controller fixture.
        """
        config = auto_controller.configure_optimizers()
        assert isinstance(config, dict)
        assert "optimizer" in config

    def test_auto_configure_optimizers_multi_stage_returns_list(
        self,
        auto_controller_two_stages: AutomaticOptimizationController,
    ) -> None:
        """Verify multi-stage configuration returns list format.

        Args:
            auto_controller_two_stages: Two-stage automatic controller fixture.
        """
        config = auto_controller_two_stages.configure_optimizers()
        assert isinstance(config, list)
        assert len(config) == 2

    def test_auto_manual_step_raises_runtime_error(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """Verify manual_step raises RuntimeError in automatic mode.

        Args:
            auto_controller: Automatic controller fixture.
        """

        def dummy_loss() -> Tensor:
            return torch.tensor(1.0)

        with pytest.raises(RuntimeError):
            auto_controller.manual_step(dummy_loss)

    def test_auto_on_epoch_end_advances_when_trigger_fires(
        self,
        auto_controller_two_stages: AutomaticOptimizationController,
    ) -> None:
        """Verify on_epoch_end advances to next stage when trigger fires.

        Args:
            auto_controller_two_stages: Two-stage automatic controller fixture.
        """
        # Epoch trigger fires at epoch 5
        auto_controller_two_stages.on_epoch_end(5, {})
        # Program should advance from stage 0 to stage 1
        assert auto_controller_two_stages._program.active_index == 1

    def test_auto_on_epoch_end_no_advance_when_trigger_silent(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """Verify on_epoch_end does not advance when trigger doesn't fire.

        Args:
            auto_controller: Automatic controller fixture.
        """
        # Stage has NoTransitionTrigger, so it never advances
        auto_controller.on_epoch_end(3, {})
        assert auto_controller._program.active_index == 0

    def test_auto_state_dict_contains_active_index(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """Verify state_dict contains the active_index.

        Args:
            auto_controller: Automatic controller fixture.
        """
        state = auto_controller.state_dict()
        assert "active_index" in state
        assert state["active_index"] == 0


# Test ManualOptimizationController
class TestManualOptimizationController:
    """Tests for ManualOptimizationController."""

    def test_manual_controller_requires_manual_optimization_is_true(
        self,
        manual_controller: ManualOptimizationController,
    ) -> None:
        """Verify that manual controller reports True for manual optimization.

        Args:
            manual_controller: Manual controller fixture.
        """
        assert manual_controller.requires_manual_optimization is True

    def test_manual_configure_optimizers_returns_list(
        self,
        manual_controller: ManualOptimizationController,
    ) -> None:
        """Verify manual controller returns list of optimizers.

        Args:
            manual_controller: Manual controller fixture.
        """
        config = manual_controller.configure_optimizers()
        assert isinstance(config, list)

    def test_manual_step_returns_tensor(
        self,
        manual_controller: ManualOptimizationController,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify manual_step returns a loss tensor.

        Args:
            manual_controller: Manual controller fixture.
            tiny_model: Tiny model fixture.
        """

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            output = tiny_model(x)
            return output.sum()

        loss = manual_controller.manual_step(loss_fn)
        assert isinstance(loss, Tensor)
        assert loss.item() is not None

    def test_manual_on_epoch_end_behavior(
        self,
        manual_controller: ManualOptimizationController,
    ) -> None:
        """Verify on_epoch_end behavior in manual mode.

        Args:
            manual_controller: Manual controller fixture.
        """
        # With NoTransitionTrigger, should stay at same index
        manual_controller.on_epoch_end(5, {})
        assert manual_controller._program.active_index == 0
