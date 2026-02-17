from __future__ import annotations

import pytest
import torch

from dlkit.nn.ffnn import (
    NormScaledConstantWidthFFNN,
    NormScaledLinearFFNN,
)


def test_norm_scaled_linear_ffnn_behaves_like_identity():
    module = NormScaledLinearFFNN(in_features=4, out_features=4, bias=False)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(4))

    rhs = torch.randn(3, 4)
    predicted = module(rhs)
    assert torch.allclose(predicted, rhs, atol=1e-6)


def test_norm_scaled_linear_ffnn_reports_norm_stats_for_zero_rhs():
    module = NormScaledLinearFFNN(in_features=3, out_features=3, bias=False, keep_stats=True)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(3))

    rhs = torch.zeros(3)
    predicted, stats = module(rhs)

    assert torch.allclose(predicted, torch.zeros_like(rhs))
    assert "norm" in stats
    assert stats["norm"].shape == (1,)
    assert torch.allclose(stats["norm"], torch.zeros_like(stats["norm"]))


def test_norm_scaled_constant_width_ffnn_is_scale_equivariant_for_positive_scalars():
    module = NormScaledConstantWidthFFNN(
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


def test_norm_scaled_linear_ffnn_rejects_integer_inputs():
    """Model should raise TypeError for integer inputs (Lightning handles casting)."""
    module = NormScaledLinearFFNN(in_features=2, out_features=2, bias=False)

    with torch.no_grad():
        module.base_model.weight.copy_(torch.eye(2))

    rhs = torch.tensor([1, 2], dtype=torch.int32)

    with pytest.raises(TypeError, match="Expected floating point tensor"):
        module(rhs)
