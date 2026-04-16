from __future__ import annotations

import pytest
import torch

from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantFFNN,
)


def test_scale_equivariant_ffnn_rejects_non_module_base_model() -> None:
    with pytest.raises(TypeError, match="base_model must be an instance"):
        ScaleEquivariantFFNN(base_model="not_a_module")  # type: ignore[arg-type]


def test_scale_equivariant_ffnn_rejects_invalid_norm() -> None:
    with pytest.raises(ValueError, match="norm must be one of"):
        ScaleEquivariantFFNN(base_model=torch.nn.Linear(2, 2), norm="l3")


def test_scale_equivariant_ffnn_rejects_non_positive_eps_gain() -> None:
    with pytest.raises(ValueError, match="eps_gain must be > 0"):
        ScaleEquivariantFFNN(base_model=torch.nn.Linear(2, 2), eps_gain=0.0)


def test_scale_equivariant_ffnn_rejects_integer_inputs() -> None:
    module = ScaleEquivariantFFNN(base_model=torch.nn.Linear(2, 2))
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
