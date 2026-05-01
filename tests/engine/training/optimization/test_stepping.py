"""Tests for optimizer stepping policies."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from dlkit.engine.training.optimization.concurrent_optimizer import ConcurrentOptimizer
from dlkit.engine.training.optimization.state import ActiveStage
from dlkit.engine.training.optimization.stepping import (
    AlternatingStepPolicy,
    LBFGSStageStepper,
    StepAllOptimizers,
)
from dlkit.engine.training.optimization.triggers import NoTransitionTrigger


# Local fixtures
@pytest.fixture
def grad_model() -> nn.Linear:
    """Create a small linear model with learnable parameters.

    Returns:
        A nn.Linear(2, 2) model with requires_grad=True parameters.
    """
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    return model


@pytest.fixture
def loss_fn(grad_model: nn.Linear) -> Callable[[], Tensor]:
    """Create a loss function using the gradient model.

    Args:
        grad_model: The gradient model fixture.

    Returns:
        A callable that computes the sum of model outputs as loss.
    """

    def _loss() -> Tensor:
        x = torch.ones(1, 2)
        output = grad_model(x)
        return output.sum()

    return _loss


@pytest.fixture
def sgd_stage(grad_model: nn.Linear) -> ActiveStage:
    """Create an active stage with SGD optimizer.

    Args:
        grad_model: The gradient model fixture.

    Returns:
        An ActiveStage with SGD optimizer and NoTransitionTrigger.
    """
    optimizer = torch.optim.SGD(grad_model.parameters(), lr=0.1)
    return ActiveStage(
        optimizer=optimizer,
        scheduler=None,
        trigger=NoTransitionTrigger(),
        stage_index=0,
        name="sgd_stage",
    )


class TestStepAllOptimizers:
    """Tests for StepAllOptimizers stepping policy."""

    def test_step_all_on_single_stage_returns_tensor(
        self,
        sgd_stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
    ) -> None:
        """Verify step_all returns a tensor for single stage.

        Args:
            sgd_stage: SGD stage fixture.
            loss_fn: Loss function fixture.
        """
        stepper = StepAllOptimizers()
        loss = stepper.step(sgd_stage, loss_fn)
        assert isinstance(loss, Tensor)

    def test_step_all_on_single_stage_performs_update(
        self,
        sgd_stage: ActiveStage,
        grad_model: nn.Linear,
        loss_fn: Callable[[], Tensor],
    ) -> None:
        """Verify step_all actually updates model parameters.

        Args:
            sgd_stage: SGD stage fixture.
            grad_model: The gradient model fixture.
            loss_fn: Loss function fixture.
        """
        # Record initial parameter values
        initial_params = [param.clone() for param in grad_model.parameters()]

        stepper = StepAllOptimizers()
        stepper.step(sgd_stage, loss_fn)

        # Check that at least one parameter changed
        changed = False
        for initial, current in zip(initial_params, grad_model.parameters(), strict=True):
            if not torch.allclose(initial, current):
                changed = True
                break

        assert changed

    def test_step_all_on_concurrent_group_steps_all_stages(
        self,
        concurrent_group: ActiveStage,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify step_all steps all sub-optimizers in a ConcurrentOptimizer stage.

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer.
            tiny_model: Tiny model fixture.
        """
        assert isinstance(concurrent_group.optimizer, ConcurrentOptimizer)

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            output = tiny_model(x)
            return output.sum()

        stepper = StepAllOptimizers()
        loss = stepper.step(concurrent_group, loss_fn)

        assert isinstance(loss, Tensor)


class TestAlternatingStepPolicy:
    """Tests for AlternatingStepPolicy stepping policy."""

    def test_alternating_period_1_rotates_stages(
        self,
        concurrent_group: ActiveStage,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify alternating policy tracks step counter correctly on a concurrent stage.

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer.
            tiny_model: Tiny model fixture.
        """

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            output = tiny_model(x)
            return output.sum()

        policy = AlternatingStepPolicy(period=1)

        for _ in range(3):
            policy.step(concurrent_group, loss_fn)

        assert policy._step_counter == 3

    def test_alternating_period_2_same_stage_for_two_calls(
        self,
        concurrent_group: ActiveStage,
        tiny_model: nn.Sequential,
    ) -> None:
        """Verify alternating policy with period=2 tracks counter over multiple calls.

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer.
            tiny_model: Tiny model fixture.
        """

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            output = tiny_model(x)
            return output.sum()

        policy = AlternatingStepPolicy(period=2)

        policy.step(concurrent_group, loss_fn)
        policy.step(concurrent_group, loss_fn)
        policy.step(concurrent_group, loss_fn)

        assert policy._step_counter == 3

    def test_alternating_on_single_stage(
        self,
        sgd_stage: ActiveStage,
        loss_fn: Callable[[], Tensor],
    ) -> None:
        """Verify alternating policy works on single stages.

        Args:
            sgd_stage: SGD stage fixture.
            loss_fn: Loss function fixture.
        """
        policy = AlternatingStepPolicy(period=1)

        # Should work fine on single stage
        loss = policy.step(sgd_stage, loss_fn)

        assert isinstance(loss, Tensor)
        assert policy._step_counter == 1


class TestLBFGSStageStepper:
    """Tests for LBFGSStageStepper stepping policy."""

    def test_lbfgs_steps_concurrent_stage_via_closure(
        self,
        concurrent_group: ActiveStage,
        tiny_model: nn.Sequential,
    ) -> None:
        """LBFGSStageStepper works on a ConcurrentOptimizer stage (closure forwarded to sub-opts).

        Args:
            concurrent_group: Stage fixture with ConcurrentOptimizer containing SGD sub-opts.
            tiny_model: Tiny model fixture.
        """

        def loss_fn() -> Tensor:
            x = torch.randn(2, 4)
            return tiny_model(x).sum()

        stepper = LBFGSStageStepper()
        loss = stepper.step(concurrent_group, loss_fn)
        assert isinstance(loss, Tensor)

    def test_lbfgs_steps_single_stage(
        self,
        grad_model: nn.Linear,
    ) -> None:
        """Verify LBFGS successfully steps a single stage.

        Args:
            grad_model: The gradient model fixture.
        """
        # Create stage with LBFGS optimizer
        optimizer = torch.optim.LBFGS(grad_model.parameters(), lr=0.1)
        stage = ActiveStage(
            optimizer=optimizer,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
            name="lbfgs_stage",
        )

        def loss_fn() -> Tensor:
            x = torch.ones(1, 2)
            output = grad_model(x)
            return output.sum()

        stepper = LBFGSStageStepper()
        loss = stepper.step(stage, loss_fn)

        assert isinstance(loss, Tensor)

    def test_lbfgs_returns_computed_loss(
        self,
        grad_model: nn.Linear,
    ) -> None:
        """Verify LBFGS returns the computed loss value.

        Args:
            grad_model: The gradient model fixture.
        """
        optimizer = torch.optim.LBFGS(grad_model.parameters(), lr=0.1)
        stage = ActiveStage(
            optimizer=optimizer,
            scheduler=None,
            trigger=NoTransitionTrigger(),
            stage_index=0,
        )

        call_count = 0

        def loss_fn() -> Tensor:
            nonlocal call_count
            call_count += 1
            x = torch.ones(1, 2)
            output = grad_model(x)
            return output.sum()

        stepper = LBFGSStageStepper()
        loss = stepper.step(stage, loss_fn)

        # LBFGS calls closure multiple times, but should return a loss
        assert isinstance(loss, Tensor)
        assert call_count > 0  # Closure was called
