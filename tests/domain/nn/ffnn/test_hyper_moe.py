from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn.ffnn import EmbeddedHyperFFNN, EmbeddedMoEFFNN, ParametricDenseBlock
from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    HyperConnection,
    HyperSequential,
    MoESequential,
    SparseMoE,
)

from ..conftest import VarianceBand

#: EmbeddedHyperFFNN/EmbeddedMoEFFNN bodies, measured ~1.48-2.20 / ~1.56-2.13 at depths 8-64.
STD_RATIO_BAND = VarianceBand(1.0, 2.5)


@pytest.mark.parametrize("model_type", [EmbeddedHyperFFNN, EmbeddedMoEFFNN])
@pytest.mark.parametrize("linear_kind", ["linear", "factorized"])
def test_embedded_hyper_moe_support_rectangular_projection(
    model_type: type[nn.Module],
    linear_kind: str,
) -> None:
    kwargs = {"num_experts": 3} if model_type is EmbeddedMoEFFNN else {}
    model = model_type(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        linear_kind=linear_kind,
        **kwargs,
    )

    out = model(torch.randn(4, 3))

    assert out.shape == (4, 5)


@pytest.mark.parametrize("model_type", [EmbeddedHyperFFNN, EmbeddedMoEFFNN])
def test_project_false_requires_square_body(model_type: type[nn.Module]) -> None:
    kwargs = {"num_experts": 3} if model_type is EmbeddedMoEFFNN else {}

    with pytest.raises(ValueError, match="project=False"):
        model_type(
            in_features=3,
            out_features=5,
            hidden_size=7,
            num_layers=2,
            project=False,
            **kwargs,
        )


@pytest.mark.parametrize("model_type", [EmbeddedHyperFFNN, EmbeddedMoEFFNN])
@pytest.mark.parametrize("linear_kind", ["linear", "factorized"])
def test_project_false_uses_identity_projections(
    model_type: type[nn.Module],
    linear_kind: str,
) -> None:
    kwargs = {"num_experts": 3} if model_type is EmbeddedMoEFFNN else {}
    model = model_type(
        in_features=4,
        out_features=4,
        hidden_size=4,
        num_layers=2,
        project=False,
        linear_kind=linear_kind,
        **kwargs,
    )

    out = model(torch.randn(5, 4))

    assert isinstance(model.embedding_layer, nn.Identity)
    assert isinstance(model.regression_layer, nn.Identity)
    assert out.shape == (5, 4)


def test_embedded_hyper_factorized_uses_factorized_body_and_projections() -> None:
    model = EmbeddedHyperFFNN(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        linear_kind="factorized",
    )

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, HyperSequential)
    for layer in model.body.layers:
        hyper_layer = cast(HyperConnection, layer)
        block = cast(ParametricDenseBlock, hyper_layer.module)
        assert isinstance(block.layer, FactorizedLinear)


def test_embedded_moe_factorized_uses_factorized_experts_and_projections() -> None:
    model = EmbeddedMoEFFNN(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
        linear_kind="factorized",
    )

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, MoESequential)
    for moe_layer in model.body.layers:
        routed_layer = cast(SparseMoE, moe_layer)
        for expert in routed_layer.experts:
            block = cast(ParametricDenseBlock, expert)
            assert isinstance(block.layer, FactorizedLinear)


def test_embedded_moe_returns_stats_after_regression() -> None:
    model = EmbeddedMoEFFNN(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
        return_stats=True,
    )

    out, stats = model(torch.randn(6, 3))

    assert out.shape == (6, 5)
    assert len(stats) == 2


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_embedded_hyper_ffnn_deep_output_std_stays_bounded(num_layers: int) -> None:
    """The standard embed -> N HyperConnection blocks -> regression network."""
    torch.manual_seed(0)
    model = EmbeddedHyperFFNN(in_features=8, out_features=8, hidden_size=8, num_layers=num_layers)
    model.eval()
    with torch.no_grad():
        x = torch.randn(256, 8)
        y = model(x)
    ratio = y.std().item() / x.std().item()
    assert STD_RATIO_BAND.low < ratio < STD_RATIO_BAND.high, (
        f"Output std diverged at {num_layers} layers: {ratio:.2f}x"
    )


@pytest.mark.parametrize("num_layers", [8, 16, 32, 64])
def test_embedded_moe_ffnn_deep_output_std_stays_bounded(num_layers: int) -> None:
    """The standard embed -> N SparseMoE blocks -> regression network."""
    torch.manual_seed(0)
    model = EmbeddedMoEFFNN(
        in_features=8,
        out_features=8,
        hidden_size=8,
        num_layers=num_layers,
        num_experts=4,
        top_k=2,
    )
    model.eval()
    with torch.no_grad():
        x = torch.randn(256, 8)
        y = model(x)
    ratio = y.std().item() / x.std().item()
    assert STD_RATIO_BAND.low < ratio < STD_RATIO_BAND.high, (
        f"Output std diverged at {num_layers} layers: {ratio:.2f}x"
    )
