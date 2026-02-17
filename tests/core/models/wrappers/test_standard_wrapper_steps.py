from __future__ import annotations

import torch
from torch import nn

from dlkit.core.datatypes.batch import Batch
from dlkit.core.models.wrappers.standard import StandardLightningWrapper
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider
from dlkit.tools.config.data_entries import Feature, Target


class _Id(nn.Module):
    def forward(self, x):  # noqa: ANN001
        # Identity: return the input tensor
        return x


def test_standard_wrapper_basic_steps(monkeypatch):  # noqa: ANN001
    # Make FactoryProvider.create_component return our simple model and loss/metric dummies
    def _fake_create(settings, ctx: BuildContext):  # noqa: ANN001
        if isinstance(settings, ModelComponentSettings):
            return _Id()
        # For metrics/loss/optimizer/scheduler, return callables/objects as needed
        return lambda a, b: torch.nn.functional.mse_loss(a, b)

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    mdl = ModelComponentSettings(
        name="_Id", module_path="tests.core.models.wrappers.test_standard_wrapper_steps"
    )
    wset = WrapperComponentSettings()
    # Create entry configs for feature and target
    entry_configs = (Feature(name="x"), Target(name="y"))
    wrapper = StandardLightningWrapper(
        model_settings=mdl, settings=wset, entry_configs=entry_configs
    )

    batch = Batch(features=(torch.ones(2, 3),), targets=(torch.zeros(2, 3),))
    out = wrapper.training_step(batch, 0)
    assert "loss" in out

    val = wrapper.validation_step(batch, 0)
    assert "val_loss" in val

    tst = wrapper.test_step(batch, 0)
    assert "test_loss" in tst
