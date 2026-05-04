from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.common.shapes import ShapeSummary
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
from dlkit.domain.nn.primitives import SkipConnection

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
    residual = residual_cls(size=4, num_layers=2)
    plain = plain_cls(size=4, num_layers=2)
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
def test_embedded_variants_support_from_shape(model_cls: type[nn.Module]) -> None:
    shape = ShapeSummary(in_shapes=((3,),), out_shapes=((2,),))
    model = cast(Any, model_cls).from_shape(shape, hidden_size=4, num_layers=2)
    x = torch.randn(4, 3)
    assert model(x).shape == (4, 2)
