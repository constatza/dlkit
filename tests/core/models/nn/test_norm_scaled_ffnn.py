from __future__ import annotations

import torch

from dlkit.nn.ffnn import (
    NormScaledConstantWidthFFNN,
    NormScaledLinearFFNN,
)
from dlkit.core.shape_specs import create_shape_spec


def _make_vector_shape(dim: int, *, output_dim: int | None = None):
    target_dim = output_dim if output_dim is not None else dim
    return create_shape_spec(
        {"rhs": (dim,), "solution": (target_dim,)},
        default_input="rhs",
        default_output="solution",
    )


def test_norm_scaled_linear_ffnn_behaves_like_identity():
    shape = _make_vector_shape(4)
    module = NormScaledLinearFFNN(unified_shape=shape, bias=False)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(4))

    rhs = torch.randn(3, 4)
    predicted = module(rhs)
    assert torch.allclose(predicted, rhs, atol=1e-6)


def test_norm_scaled_linear_ffnn_reports_norm_stats_for_zero_rhs():
    shape = _make_vector_shape(3)
    module = NormScaledLinearFFNN(unified_shape=shape, bias=False, keep_stats=True)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(3))

    rhs = torch.zeros(3)
    predicted, stats = module(rhs)

    assert torch.allclose(predicted, torch.zeros_like(rhs))
    assert "norm" in stats
    assert stats["norm"].shape == (1,)
    assert torch.allclose(stats["norm"], torch.zeros_like(stats["norm"]))


def test_norm_scaled_constant_width_ffnn_is_scale_equivariant_for_positive_scalars():
    shape = _make_vector_shape(6, output_dim=2)
    module = NormScaledConstantWidthFFNN(
        unified_shape=shape,
        hidden_size=8,
        num_layers=2,
    )

    rhs = torch.randn(5, 6)
    scale = 2.5

    base_solution = module(rhs)
    scaled_solution = module(rhs * scale)

    assert torch.allclose(scaled_solution, base_solution * scale, atol=1e-5)


def test_norm_scaled_linear_ffnn_casts_integer_inputs_to_float():
    shape = _make_vector_shape(2)
    module = NormScaledLinearFFNN(unified_shape=shape, bias=False)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(2))

    rhs = torch.tensor([1, 2], dtype=torch.int32)
    predicted = module(rhs)

    assert torch.is_floating_point(predicted)
    assert torch.allclose(predicted, rhs.to(dtype=predicted.dtype))
