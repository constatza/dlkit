from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch
from tensordict import TensorDict
from torch import nn
from torch.nn import ModuleList
from torch.optim import Adam

from dlkit.runtime.adapters.lightning.components import WrapperComponents
from dlkit.runtime.adapters.lightning.standard import StandardLightningWrapper
from dlkit.tools.config.data_entries import Feature, Target, is_feature_entry, is_target_entry
from dlkit.tools.config.model_components import (
    LossComponentSettings,
    LossInputRef,
    ModelComponentSettings,
    WrapperComponentSettings,
)


class _Id(nn.Module):
    def forward(self, x):
        return x


def _make_batch(batch_size: int = 2, feat_dim: int = 3) -> TensorDict:
    """Create a minimal TensorDict batch for wrapper step tests."""
    return TensorDict(
        {
            "features": TensorDict(
                {"x": torch.ones(batch_size, feat_dim)}, batch_size=[batch_size]
            ),
            "targets": TensorDict(
                {"y": torch.zeros(batch_size, feat_dim)}, batch_size=[batch_size]
            ),
        },
        batch_size=[batch_size],
    )


def _make_components(
    loss_fn: nn.Module | Callable[..., Any] | None = None,
    feature_names: tuple[str, ...] = (),
    target_names: tuple[str, ...] = (),
) -> WrapperComponents:
    if loss_fn is None:
        actual_loss_fn: nn.Module = nn.MSELoss()
    elif isinstance(loss_fn, nn.Module):
        actual_loss_fn = loss_fn
    else:
        # Wrap callable in a simple nn.Module subclass that preserves signature
        class _CallableModule(nn.Module):
            def __init__(self, fn: Callable[..., Any]) -> None:
                super().__init__()
                self.fn = fn
                # Copy over the original function's signature for inspect.signature()
                functools.update_wrapper(self, fn)

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.fn(*args, **kwargs)

        actual_loss_fn = _CallableModule(loss_fn)
    return WrapperComponents(
        loss_fn=actual_loss_fn,
        val_metric_routes=[],
        test_metric_routes=[],
        optimizer_factory=lambda params: Adam(params, lr=1e-3),
        scheduler_factory=None,
        feature_transforms={n: ModuleList() for n in feature_names},
        target_transforms={n: ModuleList() for n in target_names},
    )


def test_standard_wrapper_basic_steps():
    mdl = ModelComponentSettings(
        name="_Id", module_path="tests.core.models.wrappers.test_standard_wrapper_steps"
    )
    wset = WrapperComponentSettings()
    entry_configs = (Feature(name="x"), Target(name="y"))
    components = _make_components(loss_fn=nn.MSELoss(), feature_names=("x",), target_names=("y",))
    wrapper = StandardLightningWrapper(
        model_settings=mdl, settings=wset, components=components, entry_configs=entry_configs
    )

    batch = _make_batch()

    out = wrapper.training_step(batch, 0)
    assert "loss" in out

    val = wrapper.validation_step(batch, 0)
    assert "val_loss" in val

    tst = wrapper.test_step(batch, 0)
    assert "test_loss" in tst


# ============================================================================
# loss_input routing tests
# ============================================================================

_MODULE = "tests.core.models.wrappers.test_standard_wrapper_steps"


@pytest.fixture
def mdl_settings() -> ModelComponentSettings:
    """Minimal model settings pointing to the identity model in this module."""
    return ModelComponentSettings(name="_Id", module_path=_MODULE)


def _make_wrapper_with_loss(
    captured: dict,
    entry_configs: tuple,
    wrapper_settings: WrapperComponentSettings | None = None,
) -> StandardLightningWrapper:
    """Build a wrapper whose loss function captures kwargs into `captured`."""

    def _kwarg_loss(preds: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.nn.functional.mse_loss(preds, target)

    feature_names = tuple(e.name for e in entry_configs if is_feature_entry(e) and e.name)
    target_names = tuple(e.name for e in entry_configs if is_target_entry(e) and e.name)
    components = _make_components(
        loss_fn=_kwarg_loss, feature_names=feature_names, target_names=target_names
    )

    wset = wrapper_settings or WrapperComponentSettings()
    return StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
        settings=wset,
        components=components,
        entry_configs=entry_configs,
    )


def test_loss_input_str_routes_entry_as_named_kwarg() -> None:
    """loss_input='K' on a feature routes the tensor as kwarg K= to the loss function."""
    captured: dict = {}
    entry_configs = (
        Feature(name="x"),
        Feature(name="stiffness", model_input=False, loss_input="K"),
        Target(name="y"),
    )
    wrapper = _make_wrapper_with_loss(captured, entry_configs)
    K_val = torch.eye(3).unsqueeze(0).expand(2, 3, 3)
    batch = TensorDict(
        {
            "features": TensorDict({"x": torch.ones(2, 3), "stiffness": K_val}, batch_size=[2]),
            "targets": TensorDict({"y": torch.zeros(2, 3)}, batch_size=[2]),
        },
        batch_size=[2],
    )

    wrapper.training_step(batch, 0)

    assert "K" in captured
    assert captured["K"].shape == (2, 3, 3)


def test_loss_input_context_feature_excluded_from_model() -> None:
    """model_input=False, loss_input='K' entry is not passed to model but is in loss kwargs."""
    captured: dict = {}

    def _loss(preds: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.nn.functional.mse_loss(preds, target)

    entry_configs = (
        Feature(name="x"),
        Feature(name="K", model_input=False, loss_input="K"),
        Target(name="y"),
    )
    components = _make_components(loss_fn=_loss, feature_names=("x",), target_names=("y",))
    wrapper = StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
        settings=WrapperComponentSettings(),
        components=components,
        entry_configs=entry_configs,
    )

    # "K" must not appear in the model invoker's input keys
    in_keys = cast("Any", wrapper._model_invoker)._in_keys
    feature_names = [k[1] if isinstance(k, tuple) else k for k in in_keys]
    assert "K" not in feature_names
    assert "x" in feature_names

    batch = TensorDict(
        {
            "features": TensorDict(
                {"x": torch.ones(2, 3), "K": torch.eye(3).expand(2, 3, 3)}, batch_size=[2]
            ),
            "targets": TensorDict({"y": torch.zeros(2, 3)}, batch_size=[2]),
        },
        batch_size=[2],
    )

    wrapper.training_step(batch, 0)

    # Loss received K as kwarg
    assert "K" in captured


def test_duplicate_loss_input_across_entries_raises() -> None:
    """Two entries with the same loss_input value raise ValueError at construction."""
    entry_configs = (
        Feature(name="x"),
        Feature(name="K1", model_input=False, loss_input="K"),
        Feature(name="K2", model_input=False, loss_input="K"),  # duplicate
        Target(name="y"),
    )
    components = _make_components(feature_names=("x",), target_names=("y",))

    with pytest.raises(ValueError, match="Duplicate loss_input kwarg 'K'"):
        StandardLightningWrapper(
            model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
            settings=WrapperComponentSettings(),
            components=components,
            entry_configs=entry_configs,
        )


def test_loss_input_and_extra_inputs_overlap_raises() -> None:
    """Same kwarg in both DataEntry.loss_input and extra_inputs raises ValueError."""
    entry_configs = (
        Feature(name="x"),
        Feature(name="stiffness", model_input=False, loss_input="K"),
        Target(name="y"),
    )
    wset = WrapperComponentSettings(
        loss_function=LossComponentSettings(
            extra_inputs=(LossInputRef(arg="K", key="features.stiffness"),),
        )
    )
    components = _make_components(feature_names=("x",), target_names=("y",))

    with pytest.raises(ValueError, match="'K'"):
        StandardLightningWrapper(
            model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
            settings=wset,
            components=components,
            entry_configs=entry_configs,
        )


def test_loss_input_and_extra_inputs_non_overlap_merge() -> None:
    """Non-overlapping loss_input and extra_inputs both appear in the loss call."""
    captured: dict = {}

    def _loss(preds: torch.Tensor, target: torch.Tensor, **kwargs: object) -> torch.Tensor:
        captured.update(kwargs)
        return torch.nn.functional.mse_loss(preds, target)

    entry_configs = (
        Feature(name="x"),
        Feature(name="K1", model_input=False, loss_input="kwarg1"),
        Feature(name="K2", model_input=False),  # routed only via explicit extra_inputs
        Target(name="y"),
    )
    wset = WrapperComponentSettings(
        loss_function=LossComponentSettings(
            extra_inputs=(LossInputRef(arg="kwarg2", key="features.K2"),),
        )
    )
    components = _make_components(loss_fn=_loss, feature_names=("x",), target_names=("y",))
    wrapper = StandardLightningWrapper(
        model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
        settings=wset,
        components=components,
        entry_configs=entry_configs,
    )
    batch = TensorDict(
        {
            "features": TensorDict(
                {
                    "x": torch.ones(2, 3),
                    "K1": torch.ones(2, 2),
                    "K2": torch.zeros(2, 2),
                },
                batch_size=[2],
            ),
            "targets": TensorDict({"y": torch.zeros(2, 3)}, batch_size=[2]),
        },
        batch_size=[2],
    )

    wrapper.training_step(batch, 0)

    assert "kwarg1" in captured
    assert "kwarg2" in captured


def test_signature_validation_catches_missing_kwarg() -> None:
    """Build-time signature check raises ValueError when loss function lacks the kwarg."""

    def _strict_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss with no extra kwargs accepted."""
        return torch.nn.functional.mse_loss(preds, target)

    entry_configs = (
        Feature(name="x"),
        Feature(name="K", model_input=False, loss_input="K"),
        Target(name="y"),
    )
    components = _make_components(loss_fn=_strict_loss, feature_names=("x",), target_names=("y",))

    with pytest.raises(ValueError, match="no parameter named 'K'"):
        StandardLightningWrapper(
            model_settings=ModelComponentSettings(name="_Id", module_path=_MODULE),
            settings=WrapperComponentSettings(),
            components=components,
            entry_configs=entry_configs,
        )
