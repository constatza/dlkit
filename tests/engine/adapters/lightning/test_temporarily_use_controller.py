"""Unit tests for CoreLightningWrapper.temporarily_use_controller."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.engine.training.optimization.controllers import IOptimizationController
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.data_roles import DataRole
from dlkit.infrastructure.config.entry_types import ValueEntry
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)

_MODULE = "tests.engine.adapters.lightning.test_temporarily_use_controller"


class _IdentityModel(nn.Module):
    """Minimal model registered at a predictable module path for wrapper construction."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@pytest.fixture
def wrapper() -> StandardLightningWrapper:
    """Real StandardLightningWrapper wired with a single automatic-optimization stage."""
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=OptimizerPolicySettings(),
        feature_transforms={"x": ModuleList()},
        target_transforms={"y": ModuleList()},
    )
    return StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_IdentityModel", module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=(
            ValueEntry(name="x", data_role=DataRole.FEATURE),
            ValueEntry(name="y", data_role=DataRole.TARGET),
        ),
    )


@pytest.fixture
def fake_manual_controller() -> IOptimizationController:
    """Controller stub that forces manual optimization for the duration of the swap."""
    controller = MagicMock(spec=IOptimizationController)
    controller.requires_manual_optimization = True
    return controller


class TestTemporarilyUseController:
    """Verify original controller/automatic_optimization are restored on exit."""

    def test_restores_state_after_successful_block(
        self,
        wrapper: StandardLightningWrapper,
        fake_manual_controller: IOptimizationController,
    ) -> None:
        """Controller and automatic_optimization revert once the `with` block exits normally."""
        original_controller = wrapper._optimization_controller
        original_automatic_optimization = wrapper.automatic_optimization

        with wrapper.temporarily_use_controller(fake_manual_controller):
            assert wrapper._optimization_controller is fake_manual_controller
            assert wrapper.automatic_optimization is False

        assert wrapper._optimization_controller is original_controller
        assert wrapper.automatic_optimization is original_automatic_optimization

    def test_restores_state_after_exception_in_block(
        self,
        wrapper: StandardLightningWrapper,
        fake_manual_controller: IOptimizationController,
    ) -> None:
        """Controller and automatic_optimization revert even if the block raises."""
        original_controller = wrapper._optimization_controller
        original_automatic_optimization = wrapper.automatic_optimization

        with pytest.raises(RuntimeError, match="boom"):
            with wrapper.temporarily_use_controller(fake_manual_controller):
                assert wrapper._optimization_controller is fake_manual_controller
                raise RuntimeError("boom")

        assert wrapper._optimization_controller is original_controller
        assert wrapper.automatic_optimization is original_automatic_optimization
