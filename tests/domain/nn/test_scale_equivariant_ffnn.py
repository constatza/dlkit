from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch

from dlkit.common.shapes import ShapeSummary
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSimpleFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFeedForwardNN,
    ScaleEquivariantSimpleFeedForwardNN,
)
from dlkit.domain.nn.primitives import SkipConnection


def test_scale_equivariant_feed_forward_nn_is_residual() -> None:
    module = ScaleEquivariantFeedForwardNN(
        in_features=4,
        out_features=2,
        layers=[8, 8],
    )
    assert isinstance(cast(Any, module.base_model).layers[0], SkipConnection)


def test_scale_equivariant_simple_feed_forward_nn_is_plain() -> None:
    module = ScaleEquivariantSimpleFeedForwardNN(
        in_features=4,
        out_features=2,
        layers=[8, 8],
    )
    assert not isinstance(cast(Any, module.base_model).layers[0], SkipConnection)


def test_scale_equivariant_feed_forward_nn_has_no_residual_param() -> None:
    sig = inspect.signature(ScaleEquivariantFeedForwardNN.__init__)
    assert "residual" not in sig.parameters


def test_scale_equivariant_simple_feed_forward_nn_has_no_residual_param() -> None:
    sig = inspect.signature(ScaleEquivariantSimpleFeedForwardNN.__init__)
    assert "residual" not in sig.parameters


def test_scale_equivariant_feed_forward_nn_rejects_invalid_norm() -> None:
    with pytest.raises(ValueError, match="norm must be one of"):
        ScaleEquivariantFeedForwardNN(
            in_features=2,
            out_features=2,
            layers=[4, 4],
            norm="l3",
        )


def test_scale_equivariant_feed_forward_nn_rejects_non_positive_eps_gain() -> None:
    with pytest.raises(ValueError, match="eps_gain must be > 0"):
        ScaleEquivariantFeedForwardNN(
            in_features=2,
            out_features=2,
            layers=[4, 4],
            eps_gain=0.0,
        )


def test_scale_equivariant_feed_forward_nn_rejects_integer_inputs() -> None:
    module = ScaleEquivariantFeedForwardNN(
        in_features=2,
        out_features=2,
        layers=[4, 4],
    )
    rhs = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(TypeError, match="Expected floating point tensor"):
        module.forward(rhs)


def test_scale_equivariant_constant_width_ffnn_is_scale_equivariant_for_positive_scalars() -> None:
    module = ScaleEquivariantConstantWidthFFNN(
        in_features=6,
        out_features=2,
        hidden_size=8,
        num_layers=2,
    )
    rhs = torch.randn(5, 6)
    scale = 2.5
    base_solution = module(rhs)
    scaled_solution = module(rhs * scale)
    assert torch.allclose(scaled_solution, base_solution * scale, atol=1e-5)


def test_scale_equivariant_constant_width_ffnn_returns_norm_stats_when_keep_stats() -> None:
    module = ScaleEquivariantConstantWidthFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        keep_stats=True,
    )
    rhs = torch.randn(3, 4)
    out, stats = module(rhs)
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)


def test_scale_equivariant_constant_width_simple_ffnn_is_scale_equivariant_for_positive_scalars() -> (
    None
):
    module = ScaleEquivariantConstantWidthSimpleFFNN(
        in_features=6,
        out_features=2,
        hidden_size=8,
        num_layers=2,
    )
    rhs = torch.randn(5, 6)
    scale = 1.5
    base_solution = module(rhs)
    scaled_solution = module(rhs * scale)
    assert torch.allclose(scaled_solution, base_solution * scale, atol=1e-5)


@pytest.mark.parametrize(
    ("model_cls", "kwargs"),
    [
        (ScaleEquivariantConstantWidthFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantConstantWidthSimpleFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantFeedForwardNN, {"layers": [8, 8]}),
        (ScaleEquivariantSimpleFeedForwardNN, {"layers": [8, 8]}),
        (ScaleEquivariantEmbeddedSPDFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantEmbeddedSimpleSPDFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantEmbeddedSPDFactorizedFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantEmbeddedFactorizedFFNN, {"hidden_size": 8, "num_layers": 2}),
        (ScaleEquivariantEmbeddedSimpleFactorizedFFNN, {"hidden_size": 8, "num_layers": 2}),
    ],
)
def test_scale_equivariant_from_shape_ignores_duplicate_feature_kwargs(
    model_cls: type[torch.nn.Module], kwargs: dict[str, object]
) -> None:
    shape = ShapeSummary(in_shapes=((3,),), out_shapes=((2,),))

    module = cast(
        Any,
        model_cls.from_shape(
            shape,
            in_features=99,
            out_features=101,
            **kwargs,
        ),
    )

    assert module.base_model.embedding_layer.in_features == shape.in_features
    assert module.base_model.regression_layer.out_features == shape.out_features
