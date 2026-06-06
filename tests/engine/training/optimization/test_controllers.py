"""Tests for optimization controllers (automatic and manual modes)."""

from __future__ import annotations

from typing import Any, cast

from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from dlkit.common.errors import WorkflowError
from dlkit.engine.training.optimization.controllers import (
    AutomaticOptimizationController,
    ManualOptimizationController,
    _requires_manual_optimization,
)
from dlkit.engine.training.optimization.state import (
    ActiveStage,
    RunningOptimizerPolicy,
)
from dlkit.engine.training.optimization.state_repository import OptimizationStateRepository
from dlkit.engine.training.optimization.stepping import StepAllOptimizers
from dlkit.engine.training.optimization.triggers import NoTransitionTrigger


class SchedulerSpy:
    """Minimal scheduler spy for controller tests."""

    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"step_calls": self.step_calls}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_calls = state["step_calls"]


class ManualHostOptimizerSpy:
    """Minimal Lightning-style optimizer wrapper for manual host tests."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.toggle_calls = 0

    @property
    def param_groups(self) -> list[dict[str, object]]:
        return self._optimizer.param_groups

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        self._optimizer.zero_grad()

    def step(self, closure=None, **kwargs):  # noqa: ANN001,ANN003
        self.step_calls += 1
        if closure is None:
            return self._optimizer.step(**kwargs)
        return self._optimizer.step(closure=closure, **kwargs)

    @contextmanager
    def toggle_model(self, sync_grad: bool = True):  # noqa: ARG002
        self.toggle_calls += 1
        yield


class ManualHostSpy:
    """Minimal manual-optimization host exposing Lightning-like methods."""

    def __init__(self, optimizers: list[ManualHostOptimizerSpy]) -> None:
        self._optimizers = optimizers
        self.manual_backward_calls = 0

    def manual_backward(self, loss: Tensor, *args, **kwargs) -> None:  # noqa: ANN002,ANN003
        self.manual_backward_calls += 1
        loss.backward(*args, **kwargs)

    def optimizers(self, use_pl_optimizer: bool = True):  # noqa: FBT001,ARG002
        return self._optimizers


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

    def test_auto_update_learning_rate_changes_param_groups(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """update_learning_rate must mutate the active optimizer param_groups."""
        auto_controller.update_learning_rate(0.05)
        optimizer = auto_controller._program.current.optimizer
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(0.05)

    def test_auto_update_learning_rate_reflected_by_current_learning_rates(
        self,
        auto_controller: AutomaticOptimizationController,
    ) -> None:
        """current_learning_rates() must return the updated value after update_learning_rate()."""
        auto_controller.update_learning_rate(0.123)
        rates = auto_controller.current_learning_rates()
        assert all(v == pytest.approx(0.123) for v in rates.values())


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

    def test_manual_step_uses_host_manual_backward(
        self,
        manual_controller: ManualOptimizationController,
        tiny_model: nn.Sequential,
    ) -> None:
        """Manual stepping should use host.manual_backward when a host is provided."""
        wrapped = ManualHostOptimizerSpy(manual_controller._program.current.optimizer)
        host = ManualHostSpy([wrapped])

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            output = tiny_model(x)
            return output.sum()

        loss = manual_controller.manual_step(loss_fn, host)

        assert isinstance(loss, Tensor)
        assert host.manual_backward_calls == 1
        assert wrapped.zero_grad_calls == 1
        assert wrapped.step_calls == 1
        assert wrapped.toggle_calls == 1

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

    def test_manual_steps_active_stage_scheduler_before_transition(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Scheduler on the active stage must step before stage advancement."""
        optimizer_stage_0 = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        optimizer_stage_1 = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler_stage_0 = SchedulerSpy()
        scheduler_stage_1 = SchedulerSpy()
        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer_stage_0,
                    scheduler=scheduler_stage_0,
                    trigger=NoTransitionTrigger(),
                    stage_index=0,
                    scheduler_frequency=1,
                ),
                ActiveStage(
                    optimizer=optimizer_stage_1,
                    scheduler=scheduler_stage_1,
                    trigger=NoTransitionTrigger(),
                    stage_index=1,
                    scheduler_frequency=1,
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        controller.on_epoch_end(0, {})

        assert scheduler_stage_0.step_calls == 1
        assert scheduler_stage_1.step_calls == 0
        assert controller._program.active_index == 0

    def test_manual_honors_scheduler_frequency(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Manual mode should step only on epochs matching scheduler_frequency."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        scheduler = SchedulerSpy()
        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trigger=NoTransitionTrigger(),
                    stage_index=0,
                    scheduler_frequency=2,
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        controller.on_epoch_end(0, {})
        controller.on_epoch_end(1, {})
        controller.on_epoch_end(2, {})

        assert scheduler.step_calls == 1

    def test_manual_steps_current_stage_only_after_transition(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Current stage scheduler steps; newly advanced stage waits until next epoch."""
        optimizer_stage_0 = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        optimizer_stage_1 = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler_stage_0 = SchedulerSpy()
        scheduler_stage_1 = SchedulerSpy()
        trigger = NoTransitionTrigger()
        trigger.update = cast("Any", lambda epoch, metrics: True)

        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer_stage_0,
                    scheduler=scheduler_stage_0,
                    trigger=trigger,
                    stage_index=0,
                    scheduler_frequency=1,
                ),
                ActiveStage(
                    optimizer=optimizer_stage_1,
                    scheduler=scheduler_stage_1,
                    trigger=NoTransitionTrigger(),
                    stage_index=1,
                    scheduler_frequency=1,
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        controller.on_epoch_end(0, {})

        assert scheduler_stage_0.step_calls == 1
        assert scheduler_stage_1.step_calls == 0
        assert controller._program.active_index == 1

        controller.on_epoch_end(1, {})

        assert scheduler_stage_0.step_calls == 1
        assert scheduler_stage_1.step_calls == 1

    def test_manual_steps_lbfgs_scheduler(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Schedulers must still step for LBFGS-backed manual programs."""
        optimizer = torch.optim.LBFGS(tiny_model.parameters(), lr=1.0)
        scheduler = SchedulerSpy()
        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trigger=NoTransitionTrigger(),
                    stage_index=0,
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        controller.on_epoch_end(0, {})

        assert scheduler.step_calls == 1

    def test_manual_reduce_on_plateau_uses_monitored_metric(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Plateau schedulers must receive the configured monitor value."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=0
        )
        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trigger=NoTransitionTrigger(),
                    stage_index=0,
                    scheduler_monitor="val_loss",
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        controller.on_epoch_end(0, {"val_loss": 1.0})
        controller.on_epoch_end(1, {"val_loss": 1.1})

        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)

    def test_manual_update_learning_rate_changes_param_groups(
        self,
        manual_controller: ManualOptimizationController,
    ) -> None:
        """update_learning_rate must mutate the active optimizer param_groups in manual mode."""
        manual_controller.update_learning_rate(0.07)
        optimizer = manual_controller._program.current.optimizer
        for group in optimizer.param_groups:
            assert group["lr"] == pytest.approx(0.07)

    def test_manual_reduce_on_plateau_missing_metric_raises(
        self,
        tiny_model: nn.Sequential,
        repository: OptimizationStateRepository,
    ) -> None:
        """Missing scheduler monitor must raise instead of silently skipping."""
        optimizer = torch.optim.SGD(tiny_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=0
        )
        program = RunningOptimizerPolicy(
            stages=(
                ActiveStage(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trigger=NoTransitionTrigger(),
                    stage_index=0,
                    scheduler_monitor="val_loss",
                ),
            )
        )
        controller = ManualOptimizationController(program, repository, StepAllOptimizers())

        with pytest.raises(WorkflowError, match="Scheduler monitor 'val_loss' is missing"):
            controller.on_epoch_end(0, {})


# LBFGS detection tests
class SecondOrderOptimizer(torch.optim.LBFGS):
    """Subclass whose class name does NOT contain 'lbfgs'."""

    pass


class TestLBFGSDetection:
    """Tests for LBFGS detection using isinstance instead of string matching."""

    def test_requires_manual_for_single_stage_lbfgs(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify LBFGS single-stage requires manual optimization.

        Args:
            tiny_model: Tiny model fixture.
        """
        params = list(tiny_model.parameters())
        opt = torch.optim.LBFGS(params, lr=1.0)
        stage = ActiveStage(
            optimizer=opt,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        program = RunningOptimizerPolicy(stages=(stage,))
        assert _requires_manual_optimization(program) is True

    def test_requires_manual_for_lbfgs_subclass_with_different_name(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """LBFGS subclass whose name doesn't contain 'lbfgs' must still require manual mode.

        Args:
            tiny_model: Tiny model fixture.
        """
        params = list(tiny_model.parameters())
        opt = SecondOrderOptimizer(params, lr=1.0)
        stage = ActiveStage(
            optimizer=opt,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        program = RunningOptimizerPolicy(stages=(stage,))
        assert _requires_manual_optimization(program) is True

    def test_does_not_require_manual_for_adam(
        self,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify non-LBFGS optimizer doesn't require manual mode for single stage.

        Args:
            tiny_model: Tiny model fixture.
        """
        params = list(tiny_model.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        stage = ActiveStage(
            optimizer=opt,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )
        program = RunningOptimizerPolicy(stages=(stage,))
        assert _requires_manual_optimization(program) is False
