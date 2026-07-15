from __future__ import annotations

import inspect
from typing import cast

import pytest
import torch

from dlkit.domain.nn.ffnn import (
    ConstantWidthHyper,
    ConstantWidthHyperFactorized,
    ConstantWidthMoE,
    ConstantWidthMoEFactorized,
    EmbeddedHyper,
    EmbeddedHyperFactorized,
    EmbeddedMoE,
    EmbeddedMoEFactorized,
    ParametricDenseBlock,
)
from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    HyperConnection,
    HyperSequential,
    MoESequential,
    SparseMoE,
)

NEW_FACTORIZE_MODEL_TYPES = (
    ConstantWidthHyperFactorized,
    EmbeddedHyperFactorized,
    ConstantWidthMoEFactorized,
    EmbeddedMoEFactorized,
)

NEW_LINEAR_MODEL_TYPES = (
    ConstantWidthHyper,
    EmbeddedHyper,
    ConstantWidthMoE,
    EmbeddedMoE,
)


@pytest.mark.parametrize("model_type", NEW_FACTORIZE_MODEL_TYPES)
def test_new_factorized_model_constructors_do_not_expose_init_knobs(model_type: type) -> None:
    parameters = inspect.signature(model_type.__init__).parameters

    assert "mean" not in parameters
    assert "std" not in parameters
    assert "kaiming_a" not in parameters


@pytest.mark.parametrize("model_type", NEW_LINEAR_MODEL_TYPES)
def test_new_linear_model_constructors_do_not_expose_factorized_init_knobs(
    model_type: type,
) -> None:
    parameters = inspect.signature(model_type.__init__).parameters

    assert "mean" not in parameters
    assert "std" not in parameters
    assert "kaiming_a" not in parameters


@pytest.mark.parametrize(
    "model_type,kwargs",
    [
        (ConstantWidthHyperFactorized, {"num_layers": 2}),
        (ConstantWidthMoEFactorized, {"num_layers": 2, "num_experts": 3}),
    ],
)
def test_constant_width_factorized_composites_reject_rectangular_shapes(
    model_type: type,
    kwargs: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="in_features == out_features"):
        model_type(in_features=3, out_features=5, **kwargs)


@pytest.mark.parametrize(
    "model_type,kwargs",
    [
        (EmbeddedHyperFactorized, {"num_layers": 2}),
        (EmbeddedMoEFactorized, {"num_layers": 2, "num_experts": 3}),
    ],
)
def test_embedded_factorized_composites_support_rectangular_shapes(
    model_type: type,
    kwargs: dict[str, int],
) -> None:
    model = model_type(in_features=3, out_features=5, hidden_size=7, **kwargs)
    x = torch.randn(4, 3)

    out = model(x)

    assert out.shape == (4, 5)


@pytest.mark.parametrize(
    "model_type,kwargs",
    [
        (EmbeddedHyper, {"num_layers": 2}),
        (EmbeddedMoE, {"num_layers": 2, "num_experts": 3}),
    ],
)
def test_embedded_linear_composites_support_rectangular_shapes(
    model_type: type,
    kwargs: dict[str, int],
) -> None:
    model = model_type(in_features=3, out_features=5, hidden_size=7, **kwargs)
    x = torch.randn(4, 3)

    out = model(x)

    assert out.shape == (4, 5)


def test_embedded_hyper_factorized_ffnn_uses_factorized_embedding_body_and_head() -> None:
    model = EmbeddedHyperFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
    )

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, HyperSequential)
    for layer in model.body.layers:
        hyper_layer = cast(HyperConnection, layer)
        block = cast(ParametricDenseBlock, hyper_layer.module)
        assert isinstance(block.layer, FactorizedLinear)


def test_embedded_hyper_ffnn_uses_linear_embedding_body_and_head() -> None:
    model = EmbeddedHyper(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
    )

    assert isinstance(model.embedding_layer, torch.nn.Linear)
    assert not isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, torch.nn.Linear)
    assert not isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, HyperSequential)
    for layer in model.body.layers:
        hyper_layer = cast(HyperConnection, layer)
        block = cast(ParametricDenseBlock, hyper_layer.module)
        assert isinstance(block.layer, torch.nn.Linear)
        assert not isinstance(block.layer, FactorizedLinear)


def test_embedded_moe_factorized_ffnn_uses_factorized_embedding_experts_and_head() -> None:
    model = EmbeddedMoEFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
    )

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, MoESequential)
    for moe_layer in model.body.layers:
        routed_layer = cast(SparseMoE, moe_layer)
        for expert in routed_layer.experts:
            block = cast(ParametricDenseBlock, expert)
            assert isinstance(block.layer, FactorizedLinear)


def test_embedded_moe_factorized_ffnn_uses_residual_moe_stack() -> None:
    model = EmbeddedMoEFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
    )

    assert isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, MoESequential)
    assert model.body.branch_scale > 0.0
    for moe_layer in model.body.layers:
        routed_layer = cast(SparseMoE, moe_layer)
        for expert in routed_layer.experts:
            block = cast(ParametricDenseBlock, expert)
            assert isinstance(block.layer, FactorizedLinear)


def test_embedded_moe_ffnn_uses_linear_embedding_residual_stack_experts_and_head() -> None:
    model = EmbeddedMoE(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
    )

    assert isinstance(model.embedding_layer, torch.nn.Linear)
    assert not isinstance(model.embedding_layer, FactorizedLinear)
    assert isinstance(model.regression_layer, torch.nn.Linear)
    assert not isinstance(model.regression_layer, FactorizedLinear)
    assert isinstance(model.body, MoESequential)
    assert model.body.branch_scale > 0.0
    for moe_layer in model.body.layers:
        routed_layer = cast(SparseMoE, moe_layer)
        for expert in routed_layer.experts:
            block = cast(ParametricDenseBlock, expert)
            assert isinstance(block.layer, torch.nn.Linear)
            assert not isinstance(block.layer, FactorizedLinear)


def test_constant_width_linear_moe_variants_return_stats() -> None:
    x = torch.randn(6, 4)
    for model_type in (
        ConstantWidthMoE,
        ConstantWidthMoEFactorized,
    ):
        model = model_type(
            in_features=4,
            out_features=4,
            num_layers=2,
            num_experts=3,
            return_stats=True,
        )

        out, stats = model(x)

        assert out.shape == (6, 4)
        assert len(stats) == 2


def test_leaky_relu_activation_updates_factorized_base_weight_initializer_policy() -> None:
    model = EmbeddedHyperFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=1,
        activation="leaky_relu",
    )

    assert model.embedding_layer._kaiming_a == pytest.approx(0.01)
    assert model.regression_layer._kaiming_a == pytest.approx(0.01)


def test_factorized_log_scales_initialize_near_unit_scale() -> None:
    model = EmbeddedHyperFactorized(
        in_features=3,
        out_features=5,
        hidden_size=256,
        num_layers=1,
    )

    log_scale = model.embedding_layer.log_scale.detach()

    assert log_scale.mean().item() == pytest.approx(0.0, abs=0.03)
    assert log_scale.std().item() == pytest.approx(0.1, abs=0.03)
    assert torch.exp(log_scale).mean().item() == pytest.approx(1.0, abs=0.05)


def test_hyper_factorized_gradients_reach_factorized_and_mixing_parameters() -> None:
    model = EmbeddedHyperFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_lanes=3,
    )
    x = torch.randn(4, 3)

    model(x).pow(2).mean().backward()

    assert model.embedding_layer.log_scale.grad is not None
    assert model.regression_layer.base_weight.grad is not None
    first_hyper_layer = cast(HyperConnection, model.body.layers[0])
    first_block = cast(ParametricDenseBlock, first_hyper_layer.module)
    first_factorized_layer = cast(FactorizedLinear, first_block.layer)
    assert first_hyper_layer.pre_delta.grad is not None
    assert first_factorized_layer.log_scale.grad is not None


def test_moe_factorized_gradients_reach_routers_and_experts() -> None:
    model = EmbeddedMoEFactorized(
        in_features=3,
        out_features=5,
        hidden_size=7,
        num_layers=2,
        num_experts=3,
        top_k=3,
        return_stats=True,
    )
    x = torch.randn(6, 3)

    out, stats = model(x)
    (out.pow(2).mean() + sum(stat.aux_loss for stat in stats)).backward()

    first_moe_layer = cast(SparseMoE, model.body.layers[0])
    first_expert = cast(ParametricDenseBlock, first_moe_layer.experts[0])
    first_factorized_expert_layer = cast(FactorizedLinear, first_expert.layer)
    assert first_moe_layer.router.proj.weight.grad is not None
    assert first_factorized_expert_layer.log_scale.grad is not None
    assert model.embedding_layer.base_weight.grad is not None
    assert model.regression_layer.log_scale.grad is not None
    assert len(stats) == 2
