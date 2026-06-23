from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSPDFFNN,
    ScaleEquivariantEmbeddedSPDFactorizedFFNN,
    ScaleEquivariantEmbeddedSPDFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantSimpleFactorizedFFNN,
    ScaleEquivariantSimpleSPDFactorizedFFNN,
    ScaleEquivariantSimpleSPDFFNN,
    ScaleEquivariantSPDFactorizedFFNN,
    ScaleEquivariantSPDFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

# ── Fixtures ──────────────────────────────────────────────────────────────────


ShapeMapping = dict[str, tuple[int, ...]]


def _unwrap_factorized_layer(module: torch.nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


@pytest.fixture
def square_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


@pytest.fixture
def rect_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


@pytest.fixture
def nonsquare_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Mismatched (in=3, out=2) shape mappings for square-constraint errors."""
    return {"x": (3,)}, {"y": (2,)}


# ── Constant-width dense ──────────────────────────────────────────────────────


def test_scale_equivariant_constant_width_ffnn_is_scale_equivariant_for_positive_scalars() -> None:
    module = ScaleEquivariantFFNN(in_features=6, out_features=2, hidden_size=8, num_layers=2)
    rhs = torch.randn(5, 6)
    scale = 2.5
    assert torch.allclose(module(rhs * scale), module(rhs) * scale, atol=1e-5)


def test_scale_equivariant_constant_width_ffnn_returns_norm_stats_when_keep_stats() -> None:
    module = ScaleEquivariantFFNN(
        in_features=4, out_features=2, hidden_size=8, num_layers=2, keep_stats=True
    )
    out, stats = module(torch.randn(3, 4))
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)


class TestSEFFNNOptionalHiddenSize:
    def test_omit_hidden_size_when_square(self) -> None:
        m = ScaleEquivariantFFNN(in_features=4, out_features=4, num_layers=2)
        assert m(torch.randn(3, 4)).shape == (3, 4)

    def test_explicit_hidden_size_still_works(self) -> None:
        m = ScaleEquivariantFFNN(in_features=4, out_features=2, hidden_size=8, num_layers=2)
        assert m(torch.randn(3, 4)).shape == (3, 2)

    def test_raises_when_not_square_and_no_hidden_size(self) -> None:
        with pytest.raises(ValueError, match="hidden_size must be provided"):
            ScaleEquivariantFFNN(in_features=4, out_features=2, num_layers=2)


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
def test_se_embedded_spd_from_entries_requires_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    nonsquare_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = nonsquare_shapes
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, residual_cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=3)
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, plain_cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=3)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SPD_PAIRS)
def test_se_embedded_spd_from_entries_works_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, residual_cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=3)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


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
def test_se_nonembedded_spd_from_entries_requires_square(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    nonsquare_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = nonsquare_shapes
    with pytest.raises(ValueError, match="square contract"):
        cast(Any, residual_cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=2)


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
def test_se_embedded_factorized_from_entries(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, residual_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=8, num_layers=2
    )
    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


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
def test_se_nonembedded_factorized_from_entries(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, residual_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=8, num_layers=2
    )
    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize(
    "model_cls",
    [
        ScaleEquivariantEmbeddedFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ],
)
def test_se_embedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[torch.nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=1)
    body_layer = _unwrap_factorized_layer(cast(Any, model.base_model).body.blocks[0])
    assert body_layer._pos_fn is torch.exp


@pytest.mark.parametrize(
    "model_cls",
    [
        ScaleEquivariantFactorizedFFNN,
        ScaleEquivariantSimpleFactorizedFFNN,
    ],
)
def test_se_nonembedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[torch.nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    base_model = cast(Any, model.base_model)
    first_layer = base_model.first_block.layer
    body_layer = _unwrap_factorized_layer(base_model.body.blocks[0])
    assert isinstance(first_layer, FactorizedLinear)
    assert first_layer._pos_fn is torch.exp
    assert body_layer._pos_fn is torch.exp
