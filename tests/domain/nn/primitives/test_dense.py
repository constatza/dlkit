from __future__ import annotations

from typing import cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn.ffnn import FFNN, EmbeddedMoEFFNN
from dlkit.domain.nn.primitives import (
    DenseBlock,
    DenseBlockKind,
    DenseMLPBlock,
    FactorizedLinear,
    GatedDenseMLPBlock,
    HyperSequential,
    ParametricDenseBlock,
    ResidualSequential,
    SparseMoE,
    build_linear_skip_layer,
    make_dense_block,
)
from dlkit.domain.nn.primitives.skip import SkipConnection


@pytest.mark.parametrize(
    "kind",
    ["dense", "mlp", "glu", "geglu", "swiglu", "parametric"],
)
def test_make_dense_block_variants_preserve_feature_metadata(kind: DenseBlockKind) -> None:
    layer_factory = lambda n: nn.Linear(4, n)
    block = make_dense_block(
        kind,
        in_features=4,
        out_features=6,
        hidden_features=6,
        activation="gelu",
        layer_factory=layer_factory,
    )
    x = torch.randn(3, 4)

    out = block(x)

    assert out.shape == (3, 6)
    assert block.in_features == 4
    assert block.out_features == 6


def test_factorized_linear_kind_uses_factorized_linear_kernel() -> None:
    block = make_dense_block(
        "dense",
        in_features=4,
        out_features=6,
        activation="gelu",
        linear_kind="factorized",
    )

    assert isinstance(block, ParametricDenseBlock)
    assert isinstance(block.layer, FactorizedLinear)


def test_swiglu_can_use_factorized_linear_kernels() -> None:
    block = make_dense_block(
        "swiglu",
        in_features=4,
        out_features=4,
        hidden_features=8,
        linear_kind="factorized",
    )

    assert isinstance(block, GatedDenseMLPBlock)
    assert isinstance(block.value, FactorizedLinear)
    assert isinstance(block.gate, FactorizedLinear)
    assert isinstance(block.proj, FactorizedLinear)


def test_dense_mlp_block_supports_feature_last_rank3_input() -> None:
    block = DenseMLPBlock(in_features=4, out_features=5, hidden_features=7, activation="silu")
    x = torch.randn(2, 3, 4)

    out = block(x)

    assert out.shape == (2, 3, 5)


def test_gated_dense_mlp_block_uses_no_norm_by_default() -> None:
    block = GatedDenseMLPBlock(in_features=4, out_features=4, gate_activation="silu")

    assert isinstance(block.norm, nn.Identity)


@pytest.mark.parametrize("kind", ["glu", "geglu", "swiglu"])
def test_gated_dense_mlp_block_variance_ratio_is_descriptive(kind: DenseBlockKind) -> None:
    torch.manual_seed(0)
    block = make_dense_block(kind, in_features=16, out_features=16, hidden_features=32)
    x = torch.randn(1024, 16)

    with torch.no_grad():
        y = block(x)

    assert 0.01 < y.std().item() / x.std().item() < 20.0


def test_dense_block_uses_no_norm_by_default() -> None:
    block = DenseBlock(in_features=4, out_features=4)

    assert isinstance(block.norm, nn.Identity)


def test_make_dense_block_rejects_unknown_selector() -> None:
    with pytest.raises(ValueError, match="Unsupported dense block kind"):
        make_dense_block(cast(DenseBlockKind, "unknown"), in_features=4, out_features=4)


def test_parametric_selector_defaults_to_resolved_linear_factory() -> None:
    block = make_dense_block("parametric", in_features=4, out_features=4)

    assert isinstance(block, ParametricDenseBlock)
    assert isinstance(block.layer, nn.Linear)


def test_selected_block_works_inside_skip_connection() -> None:
    block = make_dense_block("swiglu", in_features=4, out_features=4, hidden_features=8)
    wrapped = SkipConnection(block, build_linear_skip_layer(block))
    x = torch.randn(5, 4)

    assert wrapped(x).shape == (5, 4)


def test_selected_block_works_inside_residual_sequential() -> None:
    block = make_dense_block("mlp", in_features=4, out_features=4, hidden_features=8)
    wrapped = ResidualSequential(block)
    x = torch.randn(5, 4)

    assert wrapped(x).shape == (5, 4)


def test_selected_block_works_as_sparse_moe_expert() -> None:
    experts = [
        make_dense_block("geglu", in_features=4, out_features=4, hidden_features=8),
        make_dense_block("geglu", in_features=4, out_features=4, hidden_features=8),
    ]
    moe = SparseMoE(in_features=4, experts=experts, top_k=1)
    x = torch.randn(5, 4)

    assert moe(x).shape == (5, 4)


def test_selected_block_works_inside_hyper_sequential() -> None:
    block = make_dense_block("mlp", in_features=4, out_features=4, hidden_features=8)
    hyper = HyperSequential(block, num_lanes=2)
    x = torch.randn(5, 4)

    assert hyper(x).shape == (5, 4)


def test_ffnn_accepts_typed_block_kind_and_records_it() -> None:
    model = FFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        block_kind="swiglu",
        linear_kind="factorized",
    )
    x = torch.randn(3, 4)

    assert model(x).shape == (3, 2)
    assert model.hyperparameters["block_kind"] == "swiglu"
    assert model.hyperparameters["linear_kind"] == "factorized"


def test_moe_accepts_typed_block_kind_for_experts() -> None:
    model = EmbeddedMoEFFNN(
        in_features=4,
        out_features=4,
        hidden_size=4,
        num_layers=1,
        num_experts=2,
        project=False,
        block_kind="geglu",
        linear_kind="factorized",
        top_k=1,
    )
    x = torch.randn(3, 4)

    assert model(x).shape == (3, 4)
    assert model.hyperparameters["block_kind"] == "geglu"
    assert model.hyperparameters["linear_kind"] == "factorized"


def test_custom_block_factory_wins_over_block_kind() -> None:
    def block_factory(**kwargs: object) -> nn.Module:
        in_features = cast(int, kwargs["in_features"])
        out_features = cast(int, kwargs["out_features"])
        return nn.Linear(in_features, out_features)

    model = FFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=1,
        block_kind="swiglu",
        block_factory=block_factory,
    )
    x = torch.randn(3, 4)

    assert model(x).shape == (3, 2)
    assert model.hyperparameters["block_kind"] == "custom"
