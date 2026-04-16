from __future__ import annotations

import torch
from torch import nn
from torch.nn import ModuleList

from dlkit.engine.adapters.lightning.factories import WrapperFactory
from dlkit.engine.adapters.lightning.wrapper_types import WrapperComponents
from dlkit.infrastructure.config import OptimizationProgramSettings
from dlkit.infrastructure.config.data_entries import Feature
from dlkit.infrastructure.config.model_components import (
    ModelComponentSettings,
    WrapperComponentSettings,
)


class _Std(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identity forward."""
        return x


class _Graph(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make_components(
    feature_names: tuple[str, ...] = (),
    target_names: tuple[str, ...] = (),
) -> WrapperComponents:
    return WrapperComponents(
        loss_fn=nn.MSELoss(),
        val_metric_routes=[],
        test_metric_routes=[],
        optimization_program_settings=OptimizationProgramSettings(),
        feature_transforms={n: ModuleList() for n in feature_names},
        target_transforms={n: ModuleList() for n in target_names},
    )


def test_wrapper_factory_detects_standard():
    mdl = ModelComponentSettings(
        name="_Std", module_path="tests.engine.adapters.lightning.test_wrapper_factory"
    )
    wrapper_type = WrapperFactory._detect_wrapper_type(mdl)
    assert wrapper_type == "standard"


def test_wrapper_factory_create_standard_wrapper():
    mdl = ModelComponentSettings(
        name="_Std", module_path="tests.engine.adapters.lightning.test_wrapper_factory"
    )
    wset = WrapperComponentSettings()
    entry_configs = (Feature("x", value=torch.zeros(4, 1)),)
    components = _make_components(feature_names=("x",))
    wrapper = WrapperFactory.create_standard_wrapper(
        model_settings=mdl,
        settings=wset,
        entry_configs=entry_configs,
        components=components,
    )
    assert wrapper is not None


def test_wrapper_factory_auto_creates_standard():
    mdl = ModelComponentSettings(
        name="_Std", module_path="tests.engine.adapters.lightning.test_wrapper_factory"
    )
    wset = WrapperComponentSettings()
    entry_configs = (Feature("x", value=torch.zeros(4, 1)),)
    components = _make_components(feature_names=("x",))
    w = WrapperFactory.create_wrapper(
        model_settings=mdl,
        settings=wset,
        wrapper_type="auto",
        entry_configs=entry_configs,
        components=components,
    )
    assert w.__class__.__name__.lower().startswith("standard")
