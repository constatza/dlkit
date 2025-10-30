from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.core.models.wrappers.standard import StandardLightningWrapper
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider


class _IdentityModel(nn.Module):
    def forward(self, **kwargs):  # noqa: ANN003
        return next(iter(kwargs.values()))


@pytest.fixture
def patch_factory(monkeypatch):  # noqa: ANN001
    def _fake_create(settings, ctx: BuildContext):  # noqa: ANN001
        if isinstance(settings, ModelComponentSettings):
            return _IdentityModel()
        return lambda a, b: torch.nn.functional.mse_loss(a, b)

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))


def _build_wrapper():
    model_settings = ModelComponentSettings(
        name="_IdentityModel",
        module_path="tests.core.models.wrappers.test_wrapper_lr_sync",
    )
    wrapper_settings = WrapperComponentSettings()
    return StandardLightningWrapper(
        model_settings=model_settings,
        settings=wrapper_settings,
        shape=None,
        entry_configs={},
    )


def test_wrapper_lr_attributes_sync_initial(patch_factory):  # noqa: ANN001
    wrapper = _build_wrapper()

    assert pytest.approx(1e-3) == wrapper.lr
    assert pytest.approx(1e-3) == wrapper.learning_rate
    assert pytest.approx(1e-3) == wrapper.hparams["lr"]
    assert pytest.approx(1e-3) == wrapper.hparams["learning_rate"]


def test_wrapper_lr_attribute_updates_optimizer_settings(patch_factory):  # noqa: ANN001
    wrapper = _build_wrapper()

    wrapper.lr = 5e-4

    assert pytest.approx(5e-4) == wrapper.lr
    assert pytest.approx(5e-4) == wrapper.learning_rate
    assert pytest.approx(5e-4) == wrapper.hparams["lr"]
    assert pytest.approx(5e-4) == wrapper.hparams["learning_rate"]
    assert pytest.approx(5e-4) == wrapper.optimizer.lr

    wrapper.learning_rate = 2e-3

    assert pytest.approx(2e-3) == wrapper.lr
    assert pytest.approx(2e-3) == wrapper.optimizer.lr
    assert pytest.approx(2e-3) == wrapper.hparams["lr"]
    assert pytest.approx(2e-3) == wrapper.hparams["learning_rate"]
