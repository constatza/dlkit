from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn import (
    SPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
    SimpleSPDFactorizedFFNN,
    SimpleSPDFFNN,
    SPDFactorizedFFNN,
)
from dlkit.domain.nn.ffnn.constrained import (
    EmbeddedParametricFFNN,
    EmbeddedSimpleParametricFFNN,
)

ShapeMapping = dict[str, tuple[int, ...]]
from dlkit.domain.nn.primitives import SkipConnection


def _dummy_factory(n: int) -> nn.Module:
    return nn.Linear(n, n)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def spd_shapes_square() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


@pytest.fixture
def spd_shapes_nonsquare() -> tuple[ShapeMapping, ShapeMapping]:
    """Mismatched (in=4, out=2) shape mappings for square-constraint errors."""
    return {"x": (4,)}, {"y": (2,)}


@pytest.fixture
def factorized_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


# ── Embedded SPD (all-SPD, square) ───────────────────────────────────────────

SPD_EMBEDDED_PAIRS = [
    (EmbeddedSPDFFNN, EmbeddedSimpleSPDFFNN),
    (EmbeddedSPDFactorizedFFNN, EmbeddedSimpleSPDFactorizedFFNN),
]

SPD_NONEMBEDDED_PAIRS = [
    (SPDFFNN, SimpleSPDFFNN),
    (SPDFactorizedFFNN, SimpleSPDFactorizedFFNN),
]

FACTORIZED_EMBEDDED_PAIRS = [
    (EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN),
]

FACTORIZED_NONEMBEDDED_PAIRS = [
    (FactorizedFFNN, SimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_EMBEDDED_PAIRS)
def test_embedded_spd_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_EMBEDDED_PAIRS)
def test_embedded_spd_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=1)
    plain = plain_cls(in_features=4, num_layers=1)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_variants_allow_zero_hidden_body_blocks(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, num_layers=0)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_from_entries_requires_square(
    model_cls: type[nn.Module],
    spd_shapes_nonsquare: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = spd_shapes_nonsquare
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, model_cls).from_entries(in_shapes, out_shapes, num_layers=3)


@pytest.mark.parametrize(
    "model_cls",
    [
        EmbeddedSPDFFNN,
        EmbeddedSimpleSPDFFNN,
        EmbeddedSPDFactorizedFFNN,
        EmbeddedSimpleSPDFactorizedFFNN,
    ],
)
def test_embedded_spd_from_entries_works_for_square(
    model_cls: type[nn.Module],
    spd_shapes_square: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = spd_shapes_square
    model = cast(Any, model_cls).from_entries(in_shapes, out_shapes, num_layers=3)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


# ── Non-embedded SPD ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_NONEMBEDDED_PAIRS)
def test_nonembedded_spd_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=3)
    plain = plain_cls(in_features=4, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 4)
    assert plain(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SPD_NONEMBEDDED_PAIRS)
def test_nonembedded_spd_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, num_layers=1)
    plain = plain_cls(in_features=4, num_layers=1)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(
    ("cls",), [(SPDFFNN,), (SimpleSPDFFNN,), (SPDFactorizedFFNN,), (SimpleSPDFactorizedFFNN,)]
)
def test_nonembedded_spd_zero_layers_has_no_body_blocks(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, num_layers=0)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 4)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize(
    "model_cls", [SPDFFNN, SimpleSPDFFNN, SPDFactorizedFFNN, SimpleSPDFactorizedFFNN]
)
def test_nonembedded_spd_from_entries_requires_square(
    model_cls: type[nn.Module],
    spd_shapes_nonsquare: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = spd_shapes_nonsquare
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, model_cls).from_entries(in_shapes, out_shapes, num_layers=2)


# ── Embedded Factorized ───────────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_EMBEDDED_PAIRS)
def test_embedded_factorized_variants_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN])
def test_embedded_factorized_from_entries(
    model_cls: type[nn.Module],
    factorized_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = factorized_shapes
    model = cast(Any, model_cls).from_entries(in_shapes, out_shapes, hidden_size=4, num_layers=2)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


# ── Non-embedded Factorized ───────────────────────────────────────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    x = torch.randn(5, 4)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_NONEMBEDDED_PAIRS)
def test_nonembedded_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    plain = plain_cls(in_features=4, out_features=2, hidden_size=8, num_layers=3)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_single_layer_has_no_body_blocks(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=2, hidden_size=8, num_layers=1)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 2)
    assert len(cast(Any, model).body.blocks) == 0


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_from_entries(
    model_cls: type[nn.Module],
    factorized_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = factorized_shapes
    model = cast(Any, model_cls).from_entries(in_shapes, out_shapes, hidden_size=8, num_layers=2)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_square_case_defaults_hidden_size(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=4, out_features=4, num_layers=2)
    x = torch.randn(3, 4)
    assert model(x).shape == (3, 4)


# ── Generic parametric builders ───────────────────────────────────────────────


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


def test_embedded_factorized_square_case_uses_in_features_as_hidden_size() -> None:
    m = EmbeddedFactorizedFFNN(in_features=4, out_features=4, num_layers=2)
    assert cast(Any, m).embedding_layer.in_features == 4
    assert cast(Any, m).embedding_layer.out_features == 4
    assert cast(Any, m).regression_layer.out_features == 4
