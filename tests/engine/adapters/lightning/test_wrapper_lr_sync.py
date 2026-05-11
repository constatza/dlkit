from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.engine.adapters.lightning.standard import StandardLightningWrapper
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.infrastructure.config import OptimizerPolicySettings
from dlkit.infrastructure.config.data_entries import Feature
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)


class _IdentityModel(nn.Module):
    def forward(self, **kwargs):
        return next(iter(kwargs.values()))


def _build_wrapper():
    model_settings = ModelComponentSettings(
        name="_IdentityModel",
        module_path="tests.engine.adapters.lightning.test_wrapper_lr_sync",
    )
    wrapper_settings = WrapperComponentSettings()
    components = WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_policy_settings=OptimizerPolicySettings(),
        feature_transforms={"x": ModuleList()},
        target_transforms={},
    )
    return StandardLightningWrapper(
        model_settings=model_settings,
        settings=wrapper_settings,
        components=components,
        entry_configs=(Feature("x", value=torch.zeros(4, 1)),),
    )


def test_wrapper_lr_attributes_sync_initial():
    wrapper = _build_wrapper()

    assert pytest.approx(1e-3) == wrapper.lr
    assert pytest.approx(1e-3) == wrapper.learning_rate
    # LR is now managed by OptimizationController, not in hparams
    # The properties delegate to the controller's program


def test_wrapper_lr_setter_delegates_to_controller():
    """Setting lr/learning_rate delegates to OptimizationController.update_learning_rate()."""
    wrapper = _build_wrapper()

    # Setter delegates to the controller — lr changes immediately
    wrapper.lr = 5e-4

    assert pytest.approx(5e-4) == wrapper.lr
    assert pytest.approx(5e-4) == wrapper.learning_rate

    # learning_rate alias also delegates through the same setter
    wrapper.learning_rate = 2e-3

    assert pytest.approx(2e-3) == wrapper.lr
    assert pytest.approx(2e-3) == wrapper.learning_rate
