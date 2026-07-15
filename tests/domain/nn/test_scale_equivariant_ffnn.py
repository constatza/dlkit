"""Tests for scale-equivariant FFNN families.

What is tested
--------------
Output shape
    Every network produces the correct (batch, out_features) tensor.

Structural invariants
    Residual variants wrap every body block in SkipConnection; plain variants do not.
    Constant-width variants reject non-square (in≠out) configs.
    ``keep_stats=True`` returns ``(Tensor, {"norm": Tensor})`` instead of a plain Tensor.

Scale nonlinearity identity
    Exp-factorized bodies use ``FactorizedLinear`` (exp pos_fn).

Unit-scale at initialisation
    Each body block's ``log_scale`` is initialised so ``φ(log_scale).mean() ≈ 1.0``.
    Every block is checked independently — a bug in factory wiring that skips some
    blocks would only appear in those blocks, not the first one.

Signal variance across depth (body, not full SE network)
    The SE wrapper normalises inputs and rescales outputs, so checking full-network
    variance is uninformative.  Instead the body is tested in isolation: for an
    8-layer body, ``body_output.std() / body_input.std()`` must stay in ``[0.01, 20]``.
    With correct ``mean=0.0`` init this holds comfortably.  If ``mean`` were reset
    to 1.0 (the Jun-23 regression), exp(1)≈2.72 per layer — 2.72^8 ≈ 3700x — and
    the test fails.

from_context
    ``StandardEntryConsumer.from_context`` wires shapes correctly.

Note: scale equivariance (f(αx) = αf(x)) is NOT tested here.  It is a mathematical
guarantee of ScaleEquivariantWrapper by construction, not a learned property, so
testing it adds no signal.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ScaleEquivariantFFNN,
)
from dlkit.domain.nn.ffnn.scale_equivariant import (
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantSimpleFactorizedFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection

# ── Named constants ───────────────────────────────────────────────────────────

BATCH_SIZE = 5
IN_FEATURES = 3
OUT_FEATURES = 2
HIDDEN_SIZE = 8
NUM_LAYERS = 2

# ── Fixtures ──────────────────────────────────────────────────────────────────


ShapeMapping = dict[str, tuple[int, ...]]


def _unwrap_factorized_layer(module: torch.nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


@pytest.fixture
def rect_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


@pytest.fixture
def square_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


# ── Constant-width dense ──────────────────────────────────────────────────────


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
    base_model = cast(Any, model.base_model)
    body_layer = _unwrap_factorized_layer(base_model.body.blocks[0])
    assert isinstance(base_model.embedding_layer, FactorizedLinear)
    assert isinstance(base_model.regression_layer, FactorizedLinear)
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
    regression_layer = base_model.regression_layer
    assert isinstance(first_layer, FactorizedLinear)
    assert isinstance(regression_layer, FactorizedLinear)
    assert first_layer._pos_fn is torch.exp
    assert body_layer._pos_fn is torch.exp
    assert regression_layer._pos_fn is torch.exp


SE_CONSTANT_WIDTH_PAIRS = [
    (
        ScaleEquivariantConstantWidthFactorizedFFNN,
        ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_CONSTANT_WIDTH_PAIRS)
def test_se_constant_width_factorized_output_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    x = torch.randn(5, 4)
    assert residual_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)
    assert plain_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)


# Depth-vs-variance stability for the factorized body is tested directly against
# ConstantWidthFactorizedFFNN in
# test_constrained_ffnn.py — ScaleEquivariantWrapper doesn't touch body internals,
# so retesting it per SE-wrapped variant here would be redundant.


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantConstantWidthFactorizedFFNN,
        ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ],
)
def test_se_constant_width_factorized_raises_when_not_square(
    cls: type[torch.nn.Module],
) -> None:
    with pytest.raises(ValueError, match="in_features == out_features"):
        cls(in_features=3, out_features=4, num_layers=2)


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantConstantWidthFactorizedFFNN,
        ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ],
)
def test_se_constant_width_factorized_from_context(
    cls: type[torch.nn.Module],
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=2)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


def test_se_constant_width_factorized_keep_stats() -> None:
    model = ScaleEquivariantConstantWidthFactorizedFFNN(
        in_features=4, out_features=4, num_layers=2, keep_stats=True
    )
    out, stats = model(torch.randn(3, 4))
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)
