"""Regression tests for graph-wrapper optimization controller selection."""

from __future__ import annotations

import pytest

from dlkit.engine.adapters.lightning.graph import GraphLightningWrapper
from dlkit.engine.training.optimization.controllers import ManualOptimizationController
from dlkit.engine.workflows.factories.component_builders import build_wrapper_components
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.infrastructure.config.optimization_stage import OptimizationStageSettings
from dlkit.infrastructure.config.optimization_trigger import TriggerSettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    StepLRSettings,
)


@pytest.fixture
def sequential_graph_policy() -> OptimizerPolicySettings:
    """Two sequential graph stages should force manual optimization."""
    return OptimizerPolicySettings(
        stages=(
            OptimizationStageSettings(
                optimizer=AdamWSettings(lr=1e-3),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
                trigger=TriggerSettings(at_epoch=5),
            ),
            OptimizationStageSettings(
                optimizer=AdamSettings(lr=1e-3),
                scheduler=StepLRSettings(step_size=1, gamma=0.5),
            ),
        )
    )


def _make_graph_wrapper(policy: OptimizerPolicySettings) -> GraphLightningWrapper:
    """Build a graph wrapper with explicit channel dimensions."""
    model_settings = ModelComponentSettings.model_validate(
        {
            "name": "GProjection",
            "module_path": "dlkit.domain.nn.graph.projection_networks",
            "hidden_size": 4,
            "in_channels": 3,
            "out_channels": 2,
        }
    )
    wrapper_settings = WrapperComponentSettings(optimizer=policy)
    components = build_wrapper_components(wrapper_settings, entry_configs=())
    return GraphLightningWrapper(
        settings=wrapper_settings,
        model_settings=model_settings,
        entry_configs=(),
        components=components,
    )


def test_graph_wrapper_uses_manual_controller_for_sequential_stages(
    sequential_graph_policy: OptimizerPolicySettings,
) -> None:
    """Sequential graph policies must use the shared manual-optimization rules."""
    wrapper = _make_graph_wrapper(sequential_graph_policy)

    assert isinstance(wrapper._optimization_controller, ManualOptimizationController)
    assert not wrapper.automatic_optimization
