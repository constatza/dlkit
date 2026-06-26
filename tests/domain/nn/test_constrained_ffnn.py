from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn import (
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
)
from dlkit.domain.nn.ffnn.constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleParametricFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

ShapeMapping = dict[str, tuple[int, ...]]


def _dummy_factory(n: int) -> nn.Module:
    return nn.Linear(n, n)


def _unwrap_factorized_layer(module: nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def factorized_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


FACTORIZED_EMBEDDED_PAIRS = [
    (EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN),
]

FACTORIZED_NONEMBEDDED_PAIRS = [
    (FactorizedFFNN, SimpleFactorizedFFNN),
]


# ── Embedded Factorized ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN])
def test_embedded_factorized_from_entries(
    model_cls: type[nn.Module],
    factorized_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = factorized_shapes
    model = cast(Any, model_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=4, num_layers=2
    )
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


# ── Non-embedded Factorized ───────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_single_layer_has_no_body_blocks(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=2, hidden_size=8, num_layers=1)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 2)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_from_entries(
    model_cls: type[nn.Module],
    factorized_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = factorized_shapes
    model = cast(Any, model_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=8, num_layers=2
    )
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_square_case_defaults_hidden_size(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=4, num_layers=2)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 4)


# ── Generic parametric builders ───────────────────────────────────────────────


def test_embedded_parametric_ffnn_is_residual() -> None:
    m = EmbeddedParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_simple_parametric_ffnn_is_plain() -> None:
    m = EmbeddedSimpleParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert not isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_parametric_ffnn_has_no_residual_param() -> None:
    sig = inspect.signature(EmbeddedParametricFFNN.__init__)
    assert "residual" not in sig.parameters


def test_embedded_factorized_square_case_uses_in_features_as_hidden_size() -> None:
    m = EmbeddedFactorizedFFNN(in_features=4, out_features=4, num_layers=2)
    assert cast(Any, m).embedding_layer.in_features == 4
    assert cast(Any, m).embedding_layer.out_features == 4
    assert cast(Any, m).regression_layer.out_features == 4


@pytest.mark.parametrize("model_cls", [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN])
def test_embedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=1)
    body_layer = _unwrap_factorized_layer(cast(Any, model).body.blocks[0])
    assert body_layer._pos_fn is torch.exp


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    first_layer = cast(Any, model).first_block.layer
    body_layer = _unwrap_factorized_layer(cast(Any, model).body.blocks[0])
    assert isinstance(first_layer, FactorizedLinear)
    assert first_layer._pos_fn is torch.exp
    assert body_layer._pos_fn is torch.exp


def test_factorized_linear_default_mean_is_zero() -> None:
    sig = inspect.signature(FactorizedLinear.__init__)
    assert sig.parameters["mean"].default == 0.0


def test_factorized_linear_unit_scale_at_default_init() -> None:
    torch.manual_seed(42)
    layer = FactorizedLinear(512, 512)
    scale = torch.exp(layer.log_scale)
    assert abs(float(scale.mean().detach()) - 1.0) < 0.3


@pytest.mark.parametrize(
    "cls",
    [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN, FactorizedFFNN, SimpleFactorizedFFNN],
)
def test_factorized_class_default_mean_is_zero(cls: type) -> None:
    sig = inspect.signature(cls.__init__)
    assert sig.parameters["mean"].default == 0.0


@pytest.fixture
def square_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


CONSTANT_WIDTH_PAIRS = [
    (ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), CONSTANT_WIDTH_PAIRS)
def test_constant_width_factorized_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    x = torch.randn(5, 4)
    assert residual_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)
    assert plain_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), CONSTANT_WIDTH_PAIRS)
def test_constant_width_factorized_all_blocks_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=4, num_layers=3)
    plain = plain_cls(in_features=4, out_features=4, num_layers=3)
    for block in cast(Any, residual).body.blocks:
        assert isinstance(block, SkipConnection)
    for block in cast(Any, plain).body.blocks:
        assert not isinstance(block, SkipConnection)


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_raises_when_not_square(cls: type[nn.Module]) -> None:
    with pytest.raises(ValueError, match="in_features.*out_features"):
        cls(in_features=3, out_features=4, num_layers=2)


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_default_activation_is_gelu(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, out_features=4, num_layers=2)
    first_block = cast(Any, model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert cast(Any, first_block).activation is F.gelu


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_from_context(
    cls: type[nn.Module],
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=2)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_body_uses_factorized_linear(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, out_features=4, num_layers=2)
    _unwrap_factorized_layer(cast(Any, model).body.blocks[0])
