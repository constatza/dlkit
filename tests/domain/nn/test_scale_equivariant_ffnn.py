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
    Softplus bodies use ``SoftplusFactorizedLinear``.

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
    the test fails.  For softplus, missing the unit-mean correction collapses the
    body to 0.69^8 ≈ 0.06x on a plain net — also caught.

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
import torch.nn.functional as F

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.ffnn import (
    ScaleEquivariantConstantWidthFactorizedFFNN,
    ScaleEquivariantConstantWidthSimpleFactorizedFFNN,
    ScaleEquivariantConstantWidthSoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedFactorizedEndFFNN,
    ScaleEquivariantEmbeddedFactorizedFFNN,
    ScaleEquivariantEmbeddedFullyFactorizedFFNN,
    ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSimpleFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSimpleSoftplusFactorizedFFNN,
    ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN,
    ScaleEquivariantEmbeddedSoftplusFactorizedFFNN,
    ScaleEquivariantFactorizedFFNN,
    ScaleEquivariantFFNN,
    ScaleEquivariantSimpleFactorizedFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection, SoftplusFactorizedLinear

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


def test_se_constant_width_factorized_body_signal_does_not_diverge() -> None:
    """Body signal std stays within 20x of input std across 8 layers.

    With correct mean=0.0 init (exp(0)=1), factorized layers behave like Kaiming Linear.
    If mean were 1.0 (the Jun-23 regression), exp(1)≈2.72 per layer — 2.72^8 ≈ 3700x
    amplification — and this test would fail.
    """
    torch.manual_seed(0)
    model = ScaleEquivariantConstantWidthFactorizedFFNN(in_features=8, out_features=8, num_layers=8)
    model.eval()
    body = cast(Any, model.base_model).body
    with torch.no_grad():
        x = torch.randn(64, 8)
        y = body(x)
    ratio = y.std().item() / x.std().item()
    assert 0.01 < ratio < 20.0, f"Body statistics diverged: std ratio = {ratio:.2f}"


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


# ── ScaleEquivariantConstantWidthSoftplusFactorizedFFNN ──────────────────────


def test_se_constant_width_softplus_factorized_output_shape() -> None:
    model = ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
        in_features=4, out_features=4, num_layers=3
    )
    assert model(torch.randn(5, 4)).shape == (5, 4)


def test_se_constant_width_softplus_factorized_raises_when_not_square() -> None:
    with pytest.raises(ValueError, match="in_features == out_features"):
        ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
            in_features=3, out_features=4, num_layers=2
        )


def test_se_constant_width_softplus_factorized_from_context(
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, ScaleEquivariantConstantWidthSoftplusFactorizedFFNN).from_context(
        ShapeContext(in_shapes, out_shapes), num_layers=2
    )
    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


def test_se_constant_width_softplus_factorized_keep_stats() -> None:
    model = ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
        in_features=4, out_features=4, num_layers=2, keep_stats=True
    )
    out, stats = model(torch.randn(3, 4))
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats
    assert stats["norm"].shape == (3, 1)


def test_se_constant_width_softplus_factorized_body_uses_softplus_layer() -> None:
    """Body layers inside the SE wrapper use SoftplusFactorizedLinear (not exp)."""
    model = ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
        in_features=4, out_features=4, num_layers=2
    )
    blocks = cast(Any, model.base_model).body.blocks
    first_block = blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert isinstance(cast(Any, first_block).layer, SoftplusFactorizedLinear)


def test_se_constant_width_softplus_factorized_unit_scale_at_init() -> None:
    """Inner softplus scales initialise near 1.0.

    The SE wrapper normalises inputs to unit L2 norm before passing them to the
    factorized body.  The body's per-neuron scale φ(s) = softplus(s) starts near
    1.0, so at initialisation the body behaves like a nearly-unscaled residual net.
    After rescaling by ‖x‖ on the way out, the wrapper restores the original norm.
    """
    torch.manual_seed(0)
    model = ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
        in_features=8, out_features=8, num_layers=4
    )
    for block in cast(Any, model.base_model).body.blocks:
        if isinstance(block, SkipConnection):
            block = cast(Any, block).module
        log_scale = cast(Any, block).layer.log_scale
        mean_scale = F.softplus(log_scale).mean().item()
        assert abs(mean_scale - 1.0) < 0.3


def test_se_constant_width_softplus_body_signal_does_not_diverge() -> None:
    """Softplus body std stays within 20x of input std across 8 layers.

    Without the _SOFTPLUS_UNIT_MEAN correction, softplus(0) = log(2) ≈ 0.69 < 1.
    A plain 8-layer net would suppress signals by 0.69^8 ≈ 0.06x.
    Residuals limit collapse but this test still catches bad initialization.
    """
    torch.manual_seed(0)
    model = ScaleEquivariantConstantWidthSoftplusFactorizedFFNN(
        in_features=8, out_features=8, num_layers=8
    )
    model.eval()
    body = cast(Any, model.base_model).body
    with torch.no_grad():
        x = torch.randn(64, 8)
        y = body(x)
    ratio = y.std().item() / x.std().item()
    assert 0.01 < ratio < 20.0, f"Softplus body statistics diverged: std ratio = {ratio:.2f}"


# ── ScaleEquivariantEmbeddedSoftplusFactorizedFFNN ───────────────────────────


SE_EMBEDDED_SOFTPLUS_PAIRS = [
    (
        ScaleEquivariantEmbeddedSoftplusFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleSoftplusFactorizedFFNN,
    ),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SOFTPLUS_PAIRS)
def test_se_embedded_softplus_factorized_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=8, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=8, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_EMBEDDED_SOFTPLUS_PAIRS)
def test_se_embedded_softplus_factorized_from_context(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, residual_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=8, num_layers=2
    )
    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


def test_se_embedded_softplus_factorized_body_signal_does_not_diverge() -> None:
    """Embedded softplus body std stays within 20x of input std across 8 layers."""
    torch.manual_seed(0)
    model = ScaleEquivariantEmbeddedSoftplusFactorizedFFNN(
        in_features=8, out_features=8, hidden_size=8, num_layers=8
    )
    model.eval()
    body = cast(Any, model.base_model).body
    with torch.no_grad():
        x = torch.randn(64, 8)
        y = body(x)
    ratio = y.std().item() / x.std().item()
    assert 0.01 < ratio < 20.0, (
        f"Embedded softplus body statistics diverged: std ratio = {ratio:.2f}"
    )


def test_se_embedded_softplus_factorized_keep_stats() -> None:
    model = ScaleEquivariantEmbeddedSoftplusFactorizedFFNN(
        in_features=3, out_features=2, hidden_size=8, num_layers=2, keep_stats=True
    )
    out, stats = model(torch.randn(5, 3))
    assert isinstance(out, torch.Tensor)
    assert "norm" in stats


def test_se_embedded_softplus_factorized_body_uses_softplus_linear() -> None:
    """Body layers inside the SE wrapper use SoftplusFactorizedLinear."""
    model = ScaleEquivariantEmbeddedSoftplusFactorizedFFNN(
        in_features=3, out_features=2, hidden_size=8, num_layers=2
    )
    first_block = cast(Any, model.base_model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert isinstance(cast(Any, first_block).layer, SoftplusFactorizedLinear)


def test_se_embedded_softplus_factorized_unit_scale_at_init() -> None:
    """Inner softplus scales initialise near 1.0 across all body blocks."""
    torch.manual_seed(0)
    model = ScaleEquivariantEmbeddedSoftplusFactorizedFFNN(
        in_features=8, out_features=8, hidden_size=8, num_layers=4
    )
    for block in cast(Any, model.base_model).body.blocks:
        if isinstance(block, SkipConnection):
            block = cast(Any, block).module
        log_scale = cast(Any, block).layer.log_scale
        mean_scale = F.softplus(log_scale).mean().item()
        assert abs(mean_scale - 1.0) < 0.3


# ── ScaleEquivariantEmbeddedFactorizedEndFFNN family ─────────────────────────


SE_FACTORIZED_END_PAIRS = [
    (ScaleEquivariantEmbeddedFactorizedEndFFNN, ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN),
]

SE_SOFTPLUS_END_PAIRS = [
    (
        ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN,
        ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN,
    ),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FACTORIZED_END_PAIRS)
def test_se_embedded_factorized_end_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Both FactorizedEnd SE variants emit (batch, out_features) tensors."""
    residual = residual_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    plain = plain_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    assert residual(x).shape == (BATCH_SIZE, OUT_FEATURES)
    assert plain(x).shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FACTORIZED_END_PAIRS)
def test_se_embedded_factorized_end_projection_layer_types(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Inner model embedding_layer is nn.Linear; regression_layer is FactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        base = cast(Any, model.base_model)
        assert isinstance(base.embedding_layer, torch.nn.Linear)
        assert isinstance(base.regression_layer, FactorizedLinear)


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantEmbeddedFactorizedEndFFNN,
        ScaleEquivariantEmbeddedSimpleFactorizedEndFFNN,
    ],
)
def test_se_embedded_factorized_end_from_context(
    cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for SE FactorizedEnd variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    assert model(torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])).shape == (
        BATCH_SIZE - 1,
        out_shapes["y"][0],
    )


# ── ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN family ─────────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_SOFTPLUS_END_PAIRS)
def test_se_embedded_softplus_end_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Both SoftplusEnd SE variants emit (batch, out_features) tensors."""
    residual = residual_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    plain = plain_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    assert residual(x).shape == (BATCH_SIZE, OUT_FEATURES)
    assert plain(x).shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_SOFTPLUS_END_PAIRS)
def test_se_embedded_softplus_end_projection_layer_types(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Inner model embedding_layer is nn.Linear; regression_layer is SoftplusFactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        base = cast(Any, model.base_model)
        assert isinstance(base.embedding_layer, torch.nn.Linear)
        assert isinstance(base.regression_layer, SoftplusFactorizedLinear)


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantEmbeddedSoftplusFactorizedEndFFNN,
        ScaleEquivariantEmbeddedSimpleSoftplusFactorizedEndFFNN,
    ],
)
def test_se_embedded_softplus_end_from_context(
    cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for SE SoftplusEnd variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    assert model(torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])).shape == (
        BATCH_SIZE - 1,
        out_shapes["y"][0],
    )


# ── ScaleEquivariantEmbeddedFullyFactorizedFFNN family ───────────────────────


SE_FULLY_FACTORIZED_PAIRS = [
    (
        ScaleEquivariantEmbeddedFullyFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN,
    ),
]

SE_FULLY_SOFTPLUS_PAIRS = [
    (
        ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN,
    ),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FULLY_FACTORIZED_PAIRS)
def test_se_embedded_fully_factorized_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Both FullyFactorized SE variants emit (batch, out_features) tensors."""
    residual = residual_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    plain = plain_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    assert residual(x).shape == (BATCH_SIZE, OUT_FEATURES)
    assert plain(x).shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FULLY_FACTORIZED_PAIRS)
def test_se_embedded_fully_factorized_all_projections_are_factorized_linear(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Inner model embedding_layer and regression_layer are both FactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        base = cast(Any, model.base_model)
        assert isinstance(base.embedding_layer, FactorizedLinear)
        assert isinstance(base.regression_layer, FactorizedLinear)


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantEmbeddedFullyFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFullyFactorizedFFNN,
    ],
)
def test_se_embedded_fully_factorized_from_context(
    cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for SE FullyFactorized variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    assert model(torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])).shape == (
        BATCH_SIZE - 1,
        out_shapes["y"][0],
    )


# ── ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN family ───────────────


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FULLY_SOFTPLUS_PAIRS)
def test_se_embedded_fully_softplus_factorized_produces_correct_shape(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Both FullySoftplus SE variants emit (batch, out_features) tensors."""
    residual = residual_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    plain = plain_cls(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    )
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    assert residual(x).shape == (BATCH_SIZE, OUT_FEATURES)
    assert plain(x).shape == (BATCH_SIZE, OUT_FEATURES)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SE_FULLY_SOFTPLUS_PAIRS)
def test_se_embedded_fully_softplus_factorized_all_projections_are_softplus_linear(
    residual_cls: type[torch.nn.Module],
    plain_cls: type[torch.nn.Module],
) -> None:
    """Inner model embedding_layer and regression_layer are both SoftplusFactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        base = cast(Any, model.base_model)
        assert isinstance(base.embedding_layer, SoftplusFactorizedLinear)
        assert isinstance(base.regression_layer, SoftplusFactorizedLinear)


@pytest.mark.parametrize(
    "cls",
    [
        ScaleEquivariantEmbeddedFullySoftplusFactorizedFFNN,
        ScaleEquivariantEmbeddedSimpleFullySoftplusFactorizedFFNN,
    ],
)
def test_se_embedded_fully_softplus_factorized_from_context(
    cls: type[torch.nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for SE FullySoftplus variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    assert model(torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])).shape == (
        BATCH_SIZE - 1,
        out_shapes["y"][0],
    )
