from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSimpleSPDFactorizedFFNN,
    ConstantWidthSimpleSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
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
)
from dlkit.domain.nn.contracts import TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import (
    ConstantWidthParametricFFNN,
    ConstantWidthSimpleParametricFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleParametricFFNN,
)
from dlkit.domain.nn.primitives import SkipConnection


def _dummy_factory(n: int) -> nn.Module:
    return nn.Linear(n, n)


CONSTANT_WIDTH_VARIANT_PAIRS = [
    (ConstantWidthSPDFFNN, ConstantWidthSimpleSPDFFNN),
    (ConstantWidthSPDFactorizedFFNN, ConstantWidthSimpleSPDFactorizedFFNN),
    (ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN),
]

EMBEDDED_VARIANT_PAIRS = [
    (EmbeddedSPDFFNN, EmbeddedSimpleSPDFFNN),
    (EmbeddedSPDFactorizedFFNN, EmbeddedSimpleSPDFactorizedFFNN),
    (EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN),
]

SCALE_EQ_CONSTANT_WIDTH_VARIANT_PAIRS = [
    (ScaleEquivariantConstantWidthSPDFFNN, ScaleEquivariantConstantWidthSimpleSPDFFNN),
    (
        ScaleEquivariantConstantWidthSPDFactorizedFFNN,
        ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN,
    ),
    (
        ScaleEquivariantConstantWidthFactorizedFFNN,
        ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ),
]

SCALE_EQ_EMBEDDED_VARIANT_PAIRS = [
    (ScaleEquivariantEmbeddedSPDFFNN, ScaleEquivariantEmbeddedSimpleSPDFFNN),
    (
        ScaleEquivariantEmbeddedSPDFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ),
    (ScaleEquivariantEmbeddedFactorizedFFNN, ScaleEquivariantEmbeddedSimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), CONSTANT_WIDTH_VARIANT_PAIRS)
def test_constant_width_constrained_variants_are_structurally_symmetric(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(size=4, num_layers=2)
    plain = plain_cls(size=4, num_layers=2)
    x = torch.randn(3, 4)

    assert residual(x).shape == (3, 4)
    assert plain(x).shape == (3, 4)
    assert isinstance(cast(Any, residual).blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), EMBEDDED_VARIANT_PAIRS)
def test_embedded_constrained_variants_are_structurally_symmetric(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(5, 3)

    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SCALE_EQ_CONSTANT_WIDTH_VARIANT_PAIRS)
def test_scale_equivariant_constant_width_constrained_variants_keep_plain_vs_residual_structure(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=2)
    plain = plain_cls(in_features=4, num_layers=2)
    x = torch.randn(3, 4)

    assert residual(x).shape == (3, 4)
    assert plain(x).shape == (3, 4)
    assert isinstance(cast(Any, residual).base_model.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).base_model.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SCALE_EQ_EMBEDDED_VARIANT_PAIRS)
def test_scale_equivariant_embedded_constrained_variants_keep_plain_vs_residual_structure(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(3, 3)

    assert residual(x).shape == (3, 2)
    assert plain(x).shape == (3, 2)
    assert isinstance(cast(Any, residual).base_model.body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).base_model.body.blocks[0], SkipConnection)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
        EmbeddedFactorizedFFNN,
        EmbeddedSimpleFactorizedFFNN,
        ScaleEquivariantEmbeddedSPDFFNN,
        ScaleEquivariantEmbeddedSimpleSPDFFNN,
        ScaleEquivariantEmbeddedSPDFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
        ScaleEquivariantEmbeddedFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ],
)
def test_embedded_variants_support_from_contract(model_cls: type[nn.Module]) -> None:
    contract = TabulaRSpec(in_shape=(3,), out_shape=(2,))
    model = cast(Any, model_cls).from_contract(contract, hidden_size=4, num_layers=2)
    x = torch.randn(4, contract.in_shape[0])
    assert model(x).shape == (4, contract.out_shape[0])


def test_constant_width_parametric_ffnn_is_residual() -> None:
    m = ConstantWidthParametricFFNN(size=8, num_layers=2, layer_factory=_dummy_factory)
    assert isinstance(cast(Any, m).blocks[0], SkipConnection)


def test_constant_width_simple_parametric_ffnn_is_plain() -> None:
    m = ConstantWidthSimpleParametricFFNN(size=8, num_layers=2, layer_factory=_dummy_factory)
    assert not isinstance(cast(Any, m).blocks[0], SkipConnection)


def test_constant_width_parametric_ffnn_has_no_residual_param() -> None:
    sig = inspect.signature(ConstantWidthParametricFFNN.__init__)
    assert "residual" not in sig.parameters


def test_constant_width_simple_parametric_ffnn_has_no_residual_param() -> None:
    sig = inspect.signature(ConstantWidthSimpleParametricFFNN.__init__)
    assert "residual" not in sig.parameters


def test_embedded_parametric_ffnn_is_residual() -> None:
    m = EmbeddedParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_simple_parametric_ffnn_is_plain() -> None:
    m = EmbeddedSimpleParametricFFNN(
        in_features=4,
        out_features=2,
        hidden_size=8,
        num_layers=2,
        layer_factory=_dummy_factory,
    )
    assert not isinstance(cast(Any, m).body.blocks[0], SkipConnection)


def test_embedded_parametric_ffnn_has_no_residual_param() -> None:
    sig = inspect.signature(EmbeddedParametricFFNN.__init__)
    assert "residual" not in sig.parameters
