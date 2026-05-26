from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch

from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFFNN,
    ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleSPDFFNN,
    ScaleEquivariantConstantWidthSPDFactorizedFFNN,
    ScaleEquivariantConstantWidthSPDFFNN,
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
def test_from_contract_ignores_duplicate_feature_kwargs(
    model_cls: type[torch.nn.Module], kwargs: dict[str, object]
) -> None:
    contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))

    module = cast(
        Any,
        model_cls.from_contract(
            contract,
            in_features=99,
            out_features=101,
            **kwargs,
        ),
    )

    assert module.base_model.embedding_layer.in_features == contract.in_shape[0]
    assert module.base_model.regression_layer.out_features == contract.out_shape[0]


class TestSEConstantWidthFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self) -> None:
        m = ScaleEquivariantConstantWidthFFNN(in_features=4, out_features=4, num_layers=2)
        x = torch.randn(3, 4)
        assert m(x).shape == (3, 4)

    def test_explicit_hidden_size_still_works(self) -> None:
        m = ScaleEquivariantConstantWidthFFNN(
            in_features=4, out_features=2, hidden_size=8, num_layers=2
        )
        x = torch.randn(3, 4)
        assert m(x).shape == (3, 2)

    def test_raises_when_not_square_and_no_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            ScaleEquivariantConstantWidthFFNN(in_features=4, out_features=2, num_layers=2)


class TestSEConstantWidthSimpleFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self) -> None:
        m = ScaleEquivariantConstantWidthSimpleFFNN(in_features=4, out_features=4, num_layers=2)
        x = torch.randn(3, 4)
        assert m(x).shape == (3, 4)

    def test_explicit_hidden_size_still_works(self) -> None:
        m = ScaleEquivariantConstantWidthSimpleFFNN(
            in_features=4, out_features=2, hidden_size=8, num_layers=2
        )
        x = torch.randn(3, 4)
        assert m(x).shape == (3, 2)

    def test_raises_when_not_square_and_no_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            ScaleEquivariantConstantWidthSimpleFFNN(in_features=4, out_features=2, num_layers=2)


@pytest.fixture
def square_input() -> torch.Tensor:
    return torch.randn(3, 4)


class TestGroupBInFeaturesRename:
    @pytest.mark.parametrize(
        "cls",
        [
            ScaleEquivariantConstantWidthSPDFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFFNN,
            ScaleEquivariantConstantWidthSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
        ],
    )
    def test_accepts_in_features(self, cls: type, square_input: torch.Tensor) -> None:
        m = cls(in_features=4, num_layers=2)
        assert m(square_input).shape == (3, 4)

    @pytest.mark.parametrize(
        "cls",
        [
            ScaleEquivariantConstantWidthSPDFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFFNN,
            ScaleEquivariantConstantWidthSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
        ],
    )
    def test_rejects_size_kwarg(self, cls: type) -> None:
        with pytest.raises(TypeError):
            cls(size=4, num_layers=2)


class TestGroupBFromContract:
    @pytest.mark.parametrize(
        "cls",
        [
            ScaleEquivariantConstantWidthSPDFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFFNN,
            ScaleEquivariantConstantWidthSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
        ],
    )
    def test_from_contract_injects_in_features(self, cls: type, square_input: torch.Tensor) -> None:
        contract = TabulaRSpec(in_shape=(4,), out_shape=(4,))
        m = cls.from_contract(contract, num_layers=2)
        assert m(square_input).shape == (square_input.shape[0], contract.in_shape[0])

    @pytest.mark.parametrize(
        "cls",
        [
            ScaleEquivariantConstantWidthSPDFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFFNN,
            ScaleEquivariantConstantWidthSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
            ScaleEquivariantConstantWidthFactorizedFFNN,
            ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
        ],
    )
    def test_from_contract_ignores_explicit_in_features_override(
        self, cls: type, square_input: torch.Tensor
    ) -> None:
        contract = TabulaRSpec(in_shape=(4,), out_shape=(4,))
        m = cls.from_contract(contract, in_features=99, num_layers=2)
        assert m(square_input).shape == (square_input.shape[0], contract.in_shape[0])
