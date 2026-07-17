from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn import ScaleEquivariantFFNN
from dlkit.domain.nn.ffnn.scale_equivariant import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedHyperFFNN,
    ScaleEquivariantEmbeddedMoEFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

from .conftest import VarianceBand

ShapeMapping = dict[str, tuple[int, ...]]


def _unwrap_factorized_layer(module: torch.nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


@pytest.fixture
def rect_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    return {"x": (3,)}, {"y": (2,)}


def test_scale_equivariant_constant_width_ffnn_returns_norm_stats_when_keep_stats() -> None:
    module = ScaleEquivariantFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        keep_stats=True,
    )

    out, stats = module(torch.randn(3, 4))

    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)


def test_scale_equivariant_ffnn_raises_without_hidden_size_for_rectangular_shape() -> None:
    with pytest.raises(ValueError, match="hidden_size must be provided"):
        ScaleEquivariantFFNN(in_features=4, out_features=2, num_layers=2)


@pytest.mark.parametrize("skip", [True, False])
def test_se_embedded_factorized_produces_correct_shape(skip: bool) -> None:
    model = ScaleEquivariantEmbeddedFactorizedFFNN(
        in_features=3,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        skip=skip,
    )

    assert model(torch.randn(5, 3)).shape == (5, 2)


def test_se_embedded_factorized_from_context(
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = rect_shapes
    model = ScaleEquivariantEmbeddedFactorizedFFNN.from_context(
        ShapeContext(in_shapes, out_shapes),
        hidden_size=8,
        num_layers=2,
    )

    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize("skip", [True, False])
def test_se_embedded_factorized_uses_factorized_layers(skip: bool) -> None:
    model = ScaleEquivariantEmbeddedFactorizedFFNN(
        in_features=3,
        out_features=2,
        hidden_size=4,
        num_layers=1,
        skip=skip,
    )
    base_model = cast(Any, model.base_model)

    assert isinstance(base_model.embedding_layer, FactorizedLinear)
    assert isinstance(base_model.regression_layer, FactorizedLinear)
    assert _unwrap_factorized_layer(base_model.body.blocks[0])._pos_fn is torch.exp


@pytest.mark.parametrize(
    ("model_type", "extra_kwargs"),
    [
        (ScaleEquivariantEmbeddedHyperFFNN, {"num_lanes": 2}),
        (ScaleEquivariantEmbeddedMoEFFNN, {"num_experts": 3}),
    ],
)
def test_se_hyper_moe_produce_correct_shape(
    model_type: type[torch.nn.Module],
    extra_kwargs: dict[str, int],
) -> None:
    model = model_type(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        **extra_kwargs,
    )

    assert model(torch.randn(4, 3)).shape == (4, 5)


@pytest.mark.parametrize(
    ("model_type", "extra_kwargs"),
    [
        (ScaleEquivariantEmbeddedHyperFFNN, {"num_lanes": 2}),
        (ScaleEquivariantEmbeddedMoEFFNN, {"num_experts": 3}),
    ],
)
def test_se_hyper_moe_are_exactly_scale_equivariant(
    model_type: type[torch.nn.Module],
    extra_kwargs: dict[str, int],
) -> None:
    """f(alpha*x) == alpha*f(x) must hold exactly through MoE routing and
    hyperconnection lane mixing: both depend only on x's normalized direction,
    which a positive scalar leaves unchanged."""
    torch.manual_seed(0)
    model = model_type(
        in_features=4,
        out_features=6,
        hidden_size=8,
        num_layers=2,
        **extra_kwargs,
    )
    model.eval()
    x = torch.randn(5, 4)
    alpha = 3.0

    with torch.no_grad():
        out = model(x)
        out_scaled = model(alpha * x)

    assert torch.allclose(out_scaled, alpha * out, atol=1e-4)


#: ScaleEquivariantEmbeddedHyperFFNN/MoEFFNN, measured ~3.56-4.47 / ~2.83-3.32
#: at depths 8-64 (higher than the unwrapped EmbeddedHyperFFNN/MoEFFNN's
#: STD_RATIO_BAND since the wrapper normalizes the input before the base
#: model runs, changing its effective operating point).
SE_HYPER_MOE_STD_RATIO_BAND = VarianceBand(1.0, 6.0)


@pytest.mark.parametrize(
    ("model_type", "extra_kwargs"),
    [
        (ScaleEquivariantEmbeddedHyperFFNN, {}),
        (ScaleEquivariantEmbeddedMoEFFNN, {"num_experts": 4}),
    ],
)
@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_se_hyper_moe_output_std_stays_bounded_with_depth(
    model_type: type[torch.nn.Module],
    extra_kwargs: dict[str, int],
    num_layers: int,
) -> None:
    torch.manual_seed(0)
    model = model_type(
        in_features=8,
        out_features=8,
        hidden_size=8,
        num_layers=num_layers,
        **extra_kwargs,
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(256, 8)
        y = model(x)
    ratio = y.std().item() / x.std().item()
    assert SE_HYPER_MOE_STD_RATIO_BAND.low < ratio < SE_HYPER_MOE_STD_RATIO_BAND.high, (
        f"{model_type.__name__} output std diverged at {num_layers} layers: {ratio:.2f}x"
    )


def test_se_embedded_moe_does_not_expose_return_stats() -> None:
    with pytest.raises(TypeError):
        ScaleEquivariantEmbeddedMoEFFNN(
            in_features=3,
            out_features=5,
            hidden_size=7,
            num_layers=2,
            num_experts=3,
            return_stats=True,  # type: ignore[call-arg]
        )
