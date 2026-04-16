"""Shared fixtures for optimization subsystem tests."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from dlkit.domain.nn.parameter_roles import ParameterRole
from dlkit.engine.training.optimization.inventory import ParameterDescriptor
from dlkit.engine.training.optimization.state import (
    ActiveConcurrentGroup,
    ActiveStage,
    RunningOptimizationProgram,
)
from dlkit.engine.training.optimization.triggers import (
    EpochTransitionTrigger,
    NoTransitionTrigger,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> nn.Sequential:
    """Two-layer linear model: Linear(4, 8) -> Linear(8, 2).

    Returns:
        A small nn.Sequential with two nn.Linear layers.
    """
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))


# ---------------------------------------------------------------------------
# Parameter descriptors
# ---------------------------------------------------------------------------


@pytest.fixture
def hidden_2d_descriptor() -> ParameterDescriptor:
    """2-D weight descriptor with HIDDEN role (Muon-eligible shape).

    Returns:
        A ParameterDescriptor for a (4, 8) weight matrix with HIDDEN role.
    """
    param = nn.Parameter(torch.randn(4, 8))
    return ParameterDescriptor(
        name="layer.weight",
        parameter=param,
        module_path="layer",
        shape=torch.Size([4, 8]),
        ndim=2,
        role=ParameterRole.HIDDEN,
    )


@pytest.fixture
def bias_1d_descriptor() -> ParameterDescriptor:
    """1-D bias descriptor with BIAS role.

    Returns:
        A ParameterDescriptor for a (8,) bias vector with BIAS role.
    """
    param = nn.Parameter(torch.randn(8))
    return ParameterDescriptor(
        name="layer.bias",
        parameter=param,
        module_path="layer",
        shape=torch.Size([8]),
        ndim=1,
        role=ParameterRole.BIAS,
    )


@pytest.fixture
def unknown_descriptor() -> ParameterDescriptor:
    """2-D weight descriptor with UNKNOWN role.

    Returns:
        A ParameterDescriptor for a (4, 8) weight matrix with UNKNOWN role.
    """
    param = nn.Parameter(torch.randn(4, 8))
    return ParameterDescriptor(
        name="other.weight",
        parameter=param,
        module_path="other",
        shape=torch.Size([4, 8]),
        ndim=2,
        role=ParameterRole.UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_optimizer(tiny_model: nn.Sequential) -> torch.optim.SGD:
    """SGD optimizer wrapping all parameters of the tiny model.

    Args:
        tiny_model: The tiny model fixture.

    Returns:
        A torch.optim.SGD instance with momentum (for state tracking).
    """
    return torch.optim.SGD(tiny_model.parameters(), lr=0.01, momentum=0.9)


# ---------------------------------------------------------------------------
# Active stages
# ---------------------------------------------------------------------------


@pytest.fixture
def no_trigger_stage(simple_optimizer: torch.optim.SGD) -> ActiveStage:
    """Single ActiveStage with a NoTransitionTrigger.

    Args:
        simple_optimizer: The SGD optimizer fixture.

    Returns:
        An ActiveStage with no transition trigger.
    """
    return ActiveStage(
        optimizer=simple_optimizer,
        scheduler=None,
        trigger=NoTransitionTrigger(),
        stage_index=0,
        name="stage_0",
    )


@pytest.fixture
def epoch_trigger_stage(tiny_model: nn.Sequential) -> ActiveStage:
    """Single ActiveStage with an EpochTransitionTrigger at epoch 5.

    Args:
        tiny_model: The tiny model fixture.

    Returns:
        An ActiveStage that transitions at epoch 5.
    """
    opt = torch.optim.SGD(tiny_model.parameters(), lr=0.01)
    return ActiveStage(
        optimizer=opt,
        scheduler=None,
        trigger=EpochTransitionTrigger(at_epoch=5),
        stage_index=0,
        name="stage_0",
    )


@pytest.fixture
def second_stage(tiny_model: nn.Sequential) -> ActiveStage:
    """Second ActiveStage with no trigger (final stage).

    Args:
        tiny_model: The tiny model fixture.

    Returns:
        An ActiveStage at index 1 with no trigger.
    """
    opt = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
    return ActiveStage(
        optimizer=opt,
        scheduler=None,
        trigger=NoTransitionTrigger(),
        stage_index=1,
        name="stage_1",
    )


@pytest.fixture
def two_stage_program(
    epoch_trigger_stage: ActiveStage,
    second_stage: ActiveStage,
) -> RunningOptimizationProgram:
    """Two-stage RunningOptimizationProgram: SGD (epoch trigger) → Adam.

    Args:
        epoch_trigger_stage: First stage with epoch trigger.
        second_stage: Second stage without trigger.

    Returns:
        A RunningOptimizationProgram with two sequential stages.
    """
    return RunningOptimizationProgram(
        stages=(epoch_trigger_stage, second_stage),
        active_index=0,
    )


@pytest.fixture
def single_stage_program(no_trigger_stage: ActiveStage) -> RunningOptimizationProgram:
    """Single-stage RunningOptimizationProgram.

    Args:
        no_trigger_stage: The only stage in the program.

    Returns:
        A RunningOptimizationProgram with one stage.
    """
    return RunningOptimizationProgram(
        stages=(no_trigger_stage,),
        active_index=0,
    )


@pytest.fixture
def concurrent_group(tiny_model: nn.Sequential) -> ActiveConcurrentGroup:
    """Concurrent group with two SGD optimizers on the same model parameters.

    Args:
        tiny_model: The tiny model fixture.

    Returns:
        An ActiveConcurrentGroup with two stages.
    """
    params = list(tiny_model.parameters())
    opt_a = torch.optim.SGD(params[:1], lr=0.01)
    opt_b = torch.optim.SGD(params[1:], lr=0.01)

    stage_a = ActiveStage(
        optimizer=opt_a,
        scheduler=None,
        trigger=NoTransitionTrigger(),
        stage_index=0,
    )
    stage_b = ActiveStage(
        optimizer=opt_b,
        scheduler=None,
        trigger=NoTransitionTrigger(),
        stage_index=1,
    )
    return ActiveConcurrentGroup(
        stages=(stage_a, stage_b),
        trigger=NoTransitionTrigger(),
        group_index=0,
    )
