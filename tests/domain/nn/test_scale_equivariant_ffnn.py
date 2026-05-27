from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch

from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSimpleFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFeedForwardNN,
    ScaleEquivariantSimpleFactorizedFFNN,
    ScaleEquivariantSimpleFeedForwardNN,
    ScaleEquivariantSimpleSPDFactorizedFFNN,
    ScaleEquivariantSimpleSPDFFNN,
    ScaleEquivariantSPDFactorizedFFNN,
    ScaleEquivariantSPDFFNN,
)
from dlkit.domain.nn.primitives import SkipConnection

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def square_contract() -> TabulaRSpec:
    return TabulaRSpec(in_shape=(4,), out_shape=(4,))


@pytest.fixture
def rect_contract() -> TabulaRSpec:
    return TabulaRSpec(in_shape=(3,), out_shape=(2,))


# ── Variable-width dense ──────────────────────────────────────────────────────


def test_scale_equivariant_feed_forward_nn_is_residual() -> None:
    module = ScaleEquivariantFeedForwardNN(in_features=4, out_features=2, layers=[8, 8])
    assert isinstance(cast(Any, module.base_model).layers[0], SkipConnection)


def test_scale_equivariant_simple_feed_forward_nn_is_plain() -> None:
    module = ScaleEquivariantSimpleFeedForwardNN(in_features=4, out_features=2, layers=[8, 8])
    assert not isinstance(cast(Any, module.base_model).layers[0], SkipConnection)


def test_scale_equivariant_feed_forward_nn_has_no_residual_param() -> None:
    sig = inspect.signature(ScaleEquivariantFeedForwardNN.__init__)
    assert "residual" not in sig.parameters


def test_scale_equivariant_simple_feed_forward_nn_has_no_residual_param() -> None:
    sig = inspect.signature(ScaleEquivariantSimpleFeedForwardNN.__init__)
    assert "residual" not in sig.parameters


def test_scale_equivariant_feed_forward_nn_rejects_invalid_norm() -> None:
    with pytest.raises(ValueError, match="norm must be one of"):
        ScaleEquivariantFeedForwardNN(in_features=2, out_features=2, layers=[4, 4], norm="l3")


def test_scale_equivariant_feed_forward_nn_rejects_non_positive_eps_gain() -> None:
    with pytest.raises(ValueError, match="eps_gain must be > 0"):
        ScaleEquivariantFeedForwardNN(in_features=2, out_features=2, layers=[4, 4], eps_gain=0.0)


def test_scale_equivariant_feed_forward_nn_rejects_integer_inputs() -> None:
    module = ScaleEquivariantFeedForwardNN(in_features=2, out_features=2, layers=[4, 4])
    rhs = torch.tensor([1, 2], dtype=torch.int32)
    with pytest.raises(TypeError, match="Expected floating point tensor"):
        module.forward(rhs)


# ── Constant-width dense ──────────────────────────────────────────────────────


def test_scale_equivariant_constant_width_ffnn_is_scale_equivariant_for_positive_scalars() -> None:
    module = ScaleEquivariantConstantWidthFFNN(
        in_features=6, out_features=2, hidden_size=8, num_layers=2
    )
    rhs = torch.randn(5, 6)
    scale = 2.5
    assert torch.allclose(module(rhs * scale), module(rhs) * scale, atol=1e-5)


def test_scale_equivariant_constant_width_ffnn_returns_norm_stats_when_keep_stats() -> None:
    module = ScaleEquivariantConstantWidthFFNN(
        in_features=4, out_features=2, hidden_size=8, num_layers=2, keep_stats=True
    )
    out, stats = module(torch.randn(3, 4))
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)


def test_scale_equivariant_constant_width_simple_ffnn_is_scale_equivariant_for_positive_scalars() -> (
    None
):
    module = ScaleEquivariantConstantWidthSimpleFFNN(
        in_features=6, out_features=2, hidden_size=8, num_layers=2
    )
    rhs = torch.randn(5, 6)
    scale = 1.5
    assert torch.allclose(module(rhs * scale), module(rhs) * scale, atol=1e-5)


class TestSEConstantWidthFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self) -> None:
        m = ScaleEquivariantConstantWidthFFNN(in_features=4, out_features=4, num_layers=2)
        assert m(torch.randn(3, 4)).shape == (3, 4)

    def test_explicit_hidden_size_still_works(self) -> None:
        m = ScaleEquivariantConstantWidthFFNN(
            in_features=4, out_features=2, hidden_size=8, num_layers=2
        )
        assert m(torch.randn(3, 4)).shape == (3, 2)

    def test_raises_when_not_square_and_no_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            ScaleEquivariantConstantWidthFFNN(in_features=4, out_features=2, num_layers=2)


class TestSEConstantWidthSimpleFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self) -> None:
        m = ScaleEquivariantConstantWidthSimpleFFNN(in_features=4, out_features=4, num_layers=2)
        assert m(torch.randn(3, 4)).shape == (3, 4)

    def test_explicit_hidden_size_still_works(self) -> None:
        m = ScaleEquivariantConstantWidthSimpleFFNN(
            in_features=4, out_features=2, hidden_size=8, num_layers=2
        )
        assert m(torch.randn(3, 4)).shape == (3, 2)

    def test_raises_when_not_square_and_no_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            ScaleEquivariantConstantWidthSimpleFFNN(in_features=4, out_features=2, num_layers=2)


# ── Embedded SPD (all-SPD, square) ───────────────────────────────────────────


SE_EMBEDDED_SPD_PAIRS = [
    (ScaleEquivariantEmbeddedSPDFFNN, ScaleEquivariantEmbeddedSimpleSPDFFNN),
    (ScaleEquivariantEmbeddedSPDFactorizedFFNN, ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN),
]

SE_NONEMBEDDED_SPD_PAIRS = [
    (ScaleEquivariantSPDFFNN, ScaleEquivariantSimpleSPDFFNN),
    (ScaleEquivariantSPDFactorizedFFNN, ScaleEquivariantSimpleSPDFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SPD_PAIRS)
def test_se_embedded_spd_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SPD_PAIRS)
def test_se_embedded_spd_from_contract_requires_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, residual_cls).from_contract(contract, num_layers=3)
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, plain_cls).from_contract(contract, num_layers=3)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SPD_PAIRS)
def test_se_embedded_spd_from_contract_works_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    square_contract: TabulaRSpec,
) -> None:
    model = cast(Any, residual_cls).from_contract(square_contract, num_layers=3)
    x = torch.randn(4, square_contract.in_shape[0])
    assert model(x).shape == (4, square_contract.out_shape[0])


# ── Non-embedded SPD ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_NONEMBEDDED_SPD_PAIRS)
def test_se_nonembedded_spd_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_NONEMBEDDED_SPD_PAIRS)
def test_se_nonembedded_spd_from_contract_requires_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, residual_cls).from_contract(contract, num_layers=2)


# ── Embedded Factorized ───────────────────────────────────────────────────────


SE_EMBEDDED_FACTORIZED_PAIRS = [
    (ScaleEquivariantEmbeddedFactorizedFFNN, ScaleEquivariantEmbeddedSimpleFactorizedFFNN),
]

SE_NONEMBEDDED_FACTORIZED_PAIRS = [
    (ScaleEquivariantFactorizedFFNN, ScaleEquivariantSimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_FACTORIZED_PAIRS)
def test_se_embedded_factorized_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=8, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=8, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_FACTORIZED_PAIRS)
def test_se_embedded_factorized_from_contract(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    rect_contract: TabulaRSpec,
) -> None:
    model = cast(Any, residual_cls).from_contract(rect_contract, hidden_size=8, num_layers=2)
    assert model(torch.randn(4, rect_contract.in_shape[0])).shape == (4, rect_contract.out_shape[0])


# ── Non-embedded Factorized ───────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_NONEMBEDDED_FACTORIZED_PAIRS)
def test_se_nonembedded_factorized_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_NONEMBEDDED_FACTORIZED_PAIRS)
def test_se_nonembedded_factorized_from_contract(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    rect_contract: TabulaRSpec,
) -> None:
    model = cast(Any, residual_cls).from_contract(rect_contract, hidden_size=8, num_layers=2)
    assert model(torch.randn(4, rect_contract.in_shape[0])).shape == (4, rect_contract.out_shape[0])
