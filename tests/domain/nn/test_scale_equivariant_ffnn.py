from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn import ScaleEquivariantFFNN
from dlkit.domain.nn.ffnn.scale_equivariant import ScaleEquivariantEmbeddedFactorizedFFNN
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

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
