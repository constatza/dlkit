from __future__ import annotations


import torch
from torch import nn

from dlkit.core.models.wrappers.factories import WrapperFactory
from dlkit.tools.config.components.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)
from dlkit.tools.config.core.context import BuildContext
from dlkit.tools.config.core.factories import FactoryProvider


class _Std(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Identity forward."""
        return x


class _Graph(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_wrapper_factory_detects_standard(monkeypatch):  # noqa: ANN001
    # FactoryProvider.create_component should return a standard-like model
    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(lambda s, c: _Std()))
    mdl = ModelComponentSettings(name="_Std", module_path="tests.core.models.wrappers.test_wrapper_factory")
    wrapper_type = WrapperFactory._detect_wrapper_type(mdl)
    assert wrapper_type == "standard"


def test_wrapper_factory_create_standard_wrapper(monkeypatch):  # noqa: ANN001
    # Patch FactoryProvider for model instantiation inside wrapper base class
    def _fake_create(settings, ctx: BuildContext):  # noqa: ANN001
        if isinstance(settings, ModelComponentSettings):
            return _Std()
        # Return simple objects for loss/metrics/optimizer etc.
        return object()

    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(_fake_create))

    mdl = ModelComponentSettings(name="_Std", module_path="tests.core.models.wrappers.test_wrapper_factory")
    wset = WrapperComponentSettings()
    wrapper = WrapperFactory.create_standard_wrapper(model_settings=mdl, settings=wset, shape=None)
    assert wrapper is not None


def test_wrapper_factory_auto_creates_standard(monkeypatch):  # noqa: ANN001
    # Detection returns standard; auto path should create standard
    monkeypatch.setattr(FactoryProvider, "create_component", staticmethod(lambda s, c: _Std()))
    mdl = ModelComponentSettings(name="_Std", module_path="tests.core.models.wrappers.test_wrapper_factory")
    wset = WrapperComponentSettings()
    w = WrapperFactory.create_wrapper(
        model_settings=mdl, settings=wset, wrapper_type="auto"
    )
    assert w.__class__.__name__.lower().startswith("standard")
