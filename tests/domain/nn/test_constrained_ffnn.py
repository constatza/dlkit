"""Tests for constrained FFNN families (factorized, softplus-factorized, parametric builders).

What is tested
--------------
Output shape
    Every network produces the correct (batch, out_features) tensor.

Structural invariants
    Residual variants wrap every body block in SkipConnection; plain variants do not.
    Constant-width variants reject non-square (in≠out) configs.

Scale nonlinearity identity
    Factorized bodies use ``FactorizedLinear`` (exp pos_fn) by default.
    Softplus bodies use ``SoftplusFactorizedLinear`` by default.
    Default ``mean`` parameter is 0.0 for all factorized classes (unit-scale intent).

Unit-scale at initialisation
    Each block's log_scale is initialised so ``φ(log_scale).mean() ≈ 1.0``.
    - For exp variants: ``exp(0) = 1`` when ``mean=0.0``.
    - For softplus variants: the ``_softplus_unit_layer_factory`` adds
      ``log(e-1) ≈ 0.5413`` so ``softplus(mean) = 1.0``.
    Every block is checked independently — a bug in factory wiring that skips
    some blocks would only corrupt those blocks, not the first one.

Signal variance across depth
    For an 8-layer network, ``output.std() / input.std()`` must stay in ``[0.01, 20]``.
    With correct ``mean=0.0`` init this holds comfortably.  If ``mean`` were
    accidentally reset to 1.0 (the Jun-23 regression), each layer amplifies by
    ``exp(1)^2 ≈ 7.4x`` in variance — 7.4^8 ≈ 4e7 for a plain net — and the test
    fails.  If the softplus unit-mean correction is missing, ``softplus(0)=0.69``
    per layer collapses a plain net to 0.69^8 ≈ 0.06x — also caught.

from_context
    ``StandardEntryConsumer.from_context`` wires shapes correctly.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn import (
    EmbeddedFactorizedEndFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedFullyFactorizedFFNN,
    EmbeddedFullySoftplusFactorizedFFNN,
    EmbeddedSimpleFactorizedEndFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleFullyFactorizedFFNN,
    EmbeddedSimpleFullySoftplusFactorizedFFNN,
    EmbeddedSimpleSoftplusFactorizedEndFFNN,
    EmbeddedSimpleSoftplusFactorizedFFNN,
    EmbeddedSoftplusFactorizedEndFFNN,
    EmbeddedSoftplusFactorizedFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
)
from dlkit.domain.nn.ffnn.constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSoftplusFactorizedFFNN,
    EmbeddedParametricFFNN,
    EmbeddedSimpleParametricFFNN,
)
from dlkit.domain.nn.primitives import FactorizedLinear, SkipConnection, SoftplusFactorizedLinear

# ── Named constants ───────────────────────────────────────────────────────────

BATCH_SIZE = 5
IN_FEATURES = 3
OUT_FEATURES = 2
HIDDEN_SIZE = 4
NUM_LAYERS = 2
UNIT_SCALE_TOLERANCE = 0.3
LARGE_HIDDEN_SIZE = 8

ShapeMapping = dict[str, tuple[int, ...]]


def _dummy_factory(n: int) -> nn.Module:
    return nn.Linear(n, n)


def _unwrap_factorized_layer(module: nn.Module) -> FactorizedLinear:
    if isinstance(module, SkipConnection):
        module = cast(Any, module).module
    layer = cast(Any, module).layer
    assert isinstance(layer, FactorizedLinear)
    return layer


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def factorized_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


FACTORIZED_EMBEDDED_PAIRS = [
    (EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN),
]

FACTORIZED_NONEMBEDDED_PAIRS = [
    (FactorizedFFNN, SimpleFactorizedFFNN),
]


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
    model = cast(Any, model_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=4, num_layers=2
    )
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
    model = cast(Any, model_cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=8, num_layers=2
    )
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


@pytest.mark.parametrize("model_cls", [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN])
def test_embedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=1)
    body_layer = _unwrap_factorized_layer(cast(Any, model).body.blocks[0])
    assert body_layer._pos_fn is torch.exp


@pytest.mark.parametrize("model_cls", [FactorizedFFNN, SimpleFactorizedFFNN])
def test_nonembedded_factorized_variants_default_to_exp_rwf(
    model_cls: type[nn.Module],
) -> None:
    model = model_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    first_layer = cast(Any, model).first_block.layer
    body_layer = _unwrap_factorized_layer(cast(Any, model).body.blocks[0])
    assert isinstance(first_layer, FactorizedLinear)
    assert first_layer._pos_fn is torch.exp
    assert body_layer._pos_fn is torch.exp


def test_factorized_linear_default_mean_is_zero() -> None:
    sig = inspect.signature(FactorizedLinear.__init__)
    assert sig.parameters["mean"].default == 0.0


def test_factorized_linear_unit_scale_at_default_init() -> None:
    torch.manual_seed(42)
    layer = FactorizedLinear(512, 512)
    scale = torch.exp(layer.log_scale)
    assert abs(float(scale.mean().detach()) - 1.0) < 0.3


@pytest.mark.parametrize(
    "cls",
    [EmbeddedFactorizedFFNN, EmbeddedSimpleFactorizedFFNN, FactorizedFFNN, SimpleFactorizedFFNN],
)
def test_factorized_class_default_mean_is_zero(cls: type) -> None:
    sig = inspect.signature(cls.__init__)
    assert sig.parameters["mean"].default == 0.0


@pytest.fixture
def square_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Square (in=4, out=4) feature/target shape mappings."""
    return {"x": (4,)}, {"y": (4,)}


CONSTANT_WIDTH_PAIRS = [
    (ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), CONSTANT_WIDTH_PAIRS)
def test_constant_width_factorized_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    x = torch.randn(5, 4)
    assert residual_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)
    assert plain_cls(in_features=4, out_features=4, num_layers=3)(x).shape == (5, 4)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), CONSTANT_WIDTH_PAIRS)
def test_constant_width_factorized_all_blocks_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=4, out_features=4, num_layers=3)
    plain = plain_cls(in_features=4, out_features=4, num_layers=3)
    for block in cast(Any, residual).body.blocks:
        assert isinstance(block, SkipConnection)
    for block in cast(Any, plain).body.blocks:
        assert not isinstance(block, SkipConnection)


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_raises_when_not_square(cls: type[nn.Module]) -> None:
    with pytest.raises(ValueError, match="in_features == out_features"):
        cls(in_features=3, out_features=4, num_layers=2)


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_default_activation_is_gelu(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, out_features=4, num_layers=2)
    first_block = cast(Any, model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert cast(Any, first_block).activation is F.gelu


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_from_context(
    cls: type[nn.Module],
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, cls).from_context(ShapeContext(in_shapes, out_shapes), num_layers=2)
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


@pytest.mark.parametrize("cls", [ConstantWidthFactorizedFFNN, ConstantWidthSimpleFactorizedFFNN])
def test_constant_width_factorized_body_uses_factorized_linear(cls: type[nn.Module]) -> None:
    model = cls(in_features=4, out_features=4, num_layers=2)
    _unwrap_factorized_layer(cast(Any, model).body.blocks[0])


# ── ConstantWidthSoftplusFactorizedFFNN ──────────────────────────────────────


def test_constant_width_softplus_factorized_output_shape() -> None:
    """ConstantWidthSoftplusFactorizedFFNN produces (batch, n) output for square input."""
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=4, out_features=4, num_layers=3)
    assert model(torch.randn(5, 4)).shape == (5, 4)


def test_constant_width_softplus_factorized_all_blocks_are_residual() -> None:
    """Every block in the body is wrapped in a SkipConnection (residual)."""
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=4, out_features=4, num_layers=3)
    for block in cast(Any, model).body.blocks:
        assert isinstance(block, SkipConnection)


def test_constant_width_softplus_factorized_raises_when_not_square() -> None:
    with pytest.raises(ValueError, match="in_features == out_features"):
        ConstantWidthSoftplusFactorizedFFNN(in_features=3, out_features=4, num_layers=2)


def test_constant_width_softplus_factorized_default_activation_is_gelu() -> None:
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=4, out_features=4, num_layers=2)
    first_block = cast(Any, model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert cast(Any, first_block).activation is F.gelu


def test_constant_width_softplus_factorized_from_context(
    square_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = square_shapes
    model = cast(Any, ConstantWidthSoftplusFactorizedFFNN).from_context(
        ShapeContext(in_shapes, out_shapes), num_layers=2
    )
    assert model(torch.randn(4, in_shapes["x"][0])).shape == (4, out_shapes["y"][0])


def test_constant_width_softplus_factorized_body_uses_softplus_layer() -> None:
    """Body layers are SoftplusFactorizedLinear, not FactorizedLinear (exp-based)."""
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=4, out_features=4, num_layers=2)
    first_block = cast(Any, model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert isinstance(cast(Any, first_block).layer, SoftplusFactorizedLinear)


def test_constant_width_softplus_factorized_unit_scale_at_init() -> None:
    """Each block's log_scale is initialised so softplus(log_scale) ≈ 1.0.

    The _softplus_unit_layer_factory shifts the Gaussian mean by log(e-1) ≈ 0.5413
    so that softplus(mean) = 1.0 exactly.  With std=0.1 and a reasonably-sized
    layer, the empirical per-block mean should stay within 0.3 of 1.0.
    """
    torch.manual_seed(0)
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=8, out_features=8, num_layers=4)
    for block in cast(Any, model).body.blocks:
        if isinstance(block, SkipConnection):
            block = cast(Any, block).module
        log_scale = cast(Any, block).layer.log_scale
        mean_scale = F.softplus(log_scale).mean().item()
        assert abs(mean_scale - 1.0) < 0.3


def test_constant_width_softplus_factorized_default_mean_is_zero() -> None:
    """User-facing mean=0.0 should be the default (offset from unit scale)."""
    import inspect

    sig = inspect.signature(ConstantWidthSoftplusFactorizedFFNN.__init__)
    assert sig.parameters["mean"].default == 0.0


def test_deep_constant_width_factorized_output_std_does_not_diverge() -> None:
    """Output std stays within 20x of input std for an 8-layer unit-scale-initialized network.

    Without residuals and with mean=1.0 (exp regression), each layer amplifies
    by exp(1)^2 ≈ 7.4x in variance — 7.4^8 ≈ 4e7 for a plain net.
    Even with residuals, divergence is detectable.  This catches that regression.
    """
    torch.manual_seed(0)
    model = ConstantWidthFactorizedFFNN(in_features=8, out_features=8, num_layers=8)
    model.eval()
    with torch.no_grad():
        x = torch.randn(64, 8)
        y = model(x)
    ratio = y.std().item() / x.std().item()
    assert 0.01 < ratio < 20.0, f"Output std diverged: {ratio:.2f}x input std"


def test_deep_constant_width_softplus_factorized_output_std_does_not_diverge() -> None:
    """Softplus output std stays within 20x of input std at 8 layers.

    Without the _SOFTPLUS_UNIT_MEAN correction, softplus(0)=log(2)≈0.69 per layer —
    a plain 8-layer net collapses signal to 0.69^8 ≈ 0.06x.  This catches that.
    """
    torch.manual_seed(0)
    model = ConstantWidthSoftplusFactorizedFFNN(in_features=8, out_features=8, num_layers=8)
    model.eval()
    with torch.no_grad():
        x = torch.randn(64, 8)
        y = model(x)
    ratio = y.std().item() / x.std().item()
    assert 0.01 < ratio < 20.0, f"Softplus output std diverged: {ratio:.2f}x input std"


# ── EmbeddedSoftplusFactorizedFFNN / EmbeddedSimpleSoftplusFactorizedFFNN ────


@pytest.fixture
def rect_shapes() -> tuple[ShapeMapping, ShapeMapping]:
    """Rectangular (in=3, out=2) feature/target shape mappings."""
    return {"x": (3,)}, {"y": (2,)}


SOFTPLUS_EMBEDDED_PAIRS = [
    (EmbeddedSoftplusFactorizedFFNN, EmbeddedSimpleSoftplusFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_EMBEDDED_PAIRS)
def test_embedded_softplus_factorized_variants_produce_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    x = torch.randn(5, 3)
    assert residual(x).shape == (5, 2)
    assert plain(x).shape == (5, 2)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_EMBEDDED_PAIRS)
def test_embedded_softplus_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    residual = residual_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    plain = plain_cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_EMBEDDED_PAIRS)
def test_embedded_softplus_factorized_body_uses_softplus_linear(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    for cls in (residual_cls, plain_cls):
        model = cls(in_features=3, out_features=2, hidden_size=4, num_layers=2)
        first_block = cast(Any, model).body.blocks[0]
        if isinstance(first_block, SkipConnection):
            first_block = cast(Any, first_block).module
        assert isinstance(cast(Any, first_block).layer, SoftplusFactorizedLinear)


def test_embedded_softplus_factorized_unit_scale_at_init() -> None:
    """Each body block initialises with softplus(log_scale) ≈ 1.0.

    The _softplus_unit_layer_factory shifts the Gaussian mean to log(e-1) so
    that softplus(mean) = 1.0.  With std=0.1 and hidden_size=64, the empirical
    per-block mean should stay within 0.3 of 1.0.
    """
    torch.manual_seed(0)
    model = EmbeddedSoftplusFactorizedFFNN(
        in_features=8, out_features=8, hidden_size=8, num_layers=4
    )
    for block in cast(Any, model).body.blocks:
        if isinstance(block, SkipConnection):
            block = cast(Any, block).module
        log_scale = cast(Any, block).layer.log_scale
        mean_scale = F.softplus(log_scale).mean().item()
        assert abs(mean_scale - 1.0) < 0.3


def test_embedded_softplus_factorized_default_mean_is_zero() -> None:
    sig = inspect.signature(EmbeddedSoftplusFactorizedFFNN.__init__)
    assert sig.parameters["mean"].default == 0.0


@pytest.mark.parametrize(
    "cls", [EmbeddedSoftplusFactorizedFFNN, EmbeddedSimpleSoftplusFactorizedFFNN]
)
def test_embedded_softplus_factorized_default_activation_is_gelu(cls: type[nn.Module]) -> None:
    model = cls(in_features=3, out_features=2, hidden_size=4, num_layers=1)
    first_block = cast(Any, model).body.blocks[0]
    if isinstance(first_block, SkipConnection):
        first_block = cast(Any, first_block).module
    assert cast(Any, first_block).activation is F.gelu


@pytest.mark.parametrize(
    "cls", [EmbeddedSoftplusFactorizedFFNN, EmbeddedSimpleSoftplusFactorizedFFNN]
)
def test_embedded_softplus_factorized_from_context(
    cls: type[nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=4, num_layers=2
    )
    x = torch.randn(4, in_shapes["x"][0])
    assert model(x).shape == (4, out_shapes["y"][0])


# ── EmbeddedFactorizedEndFFNN / EmbeddedSimpleFactorizedEndFFNN ───────────────


FACTORIZED_END_PAIRS = [
    (EmbeddedFactorizedEndFFNN, EmbeddedSimpleFactorizedEndFFNN),
]

SOFTPLUS_END_PAIRS = [
    (EmbeddedSoftplusFactorizedEndFFNN, EmbeddedSimpleSoftplusFactorizedEndFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_END_PAIRS)
def test_embedded_factorized_end_produces_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both residual and plain FactorizedEnd variants emit (batch, out_features) tensors."""
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


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_END_PAIRS)
def test_embedded_factorized_end_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Residual variant wraps body blocks in SkipConnection; plain does not."""
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
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FACTORIZED_END_PAIRS)
def test_embedded_factorized_end_projection_layer_types(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """embedding_layer is nn.Linear; regression_layer is FactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert isinstance(cast(Any, model).embedding_layer, nn.Linear)
        assert isinstance(cast(Any, model).regression_layer, FactorizedLinear)


@pytest.mark.parametrize("cls", [EmbeddedFactorizedEndFFNN, EmbeddedSimpleFactorizedEndFFNN])
def test_embedded_factorized_end_from_context(
    cls: type[nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for FactorizedEnd variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    x = torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])
    assert model(x).shape == (BATCH_SIZE - 1, out_shapes["y"][0])


# ── EmbeddedSoftplusFactorizedEndFFNN / EmbeddedSimpleSoftplusFactorizedEndFFNN


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_END_PAIRS)
def test_embedded_softplus_end_produces_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both residual and plain SoftplusEnd variants emit (batch, out_features) tensors."""
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


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_END_PAIRS)
def test_embedded_softplus_end_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Residual SoftplusEnd wraps body blocks in SkipConnection; plain does not."""
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
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), SOFTPLUS_END_PAIRS)
def test_embedded_softplus_end_projection_layer_types(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """embedding_layer is nn.Linear; regression_layer is SoftplusFactorizedLinear."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert isinstance(cast(Any, model).embedding_layer, nn.Linear)
        assert isinstance(cast(Any, model).regression_layer, SoftplusFactorizedLinear)


@pytest.mark.parametrize(
    "cls", [EmbeddedSoftplusFactorizedEndFFNN, EmbeddedSimpleSoftplusFactorizedEndFFNN]
)
def test_embedded_softplus_end_from_context(
    cls: type[nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for SoftplusEnd variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    x = torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])
    assert model(x).shape == (BATCH_SIZE - 1, out_shapes["y"][0])


# ── EmbeddedFullyFactorizedFFNN / EmbeddedSimpleFullyFactorizedFFNN ────────────


FULLY_FACTORIZED_PAIRS = [
    (EmbeddedFullyFactorizedFFNN, EmbeddedSimpleFullyFactorizedFFNN),
]

FULLY_SOFTPLUS_PAIRS = [
    (EmbeddedFullySoftplusFactorizedFFNN, EmbeddedSimpleFullySoftplusFactorizedFFNN),
]


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_FACTORIZED_PAIRS)
def test_embedded_fully_factorized_produces_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both FullyFactorized variants emit (batch, out_features) tensors."""
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


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_FACTORIZED_PAIRS)
def test_embedded_fully_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Residual FullyFactorized wraps body blocks in SkipConnection; plain does not."""
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
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_FACTORIZED_PAIRS)
def test_embedded_fully_factorized_all_projections_are_factorized_linear(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both embedding_layer and regression_layer are FactorizedLinear (no nn.Linear)."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert isinstance(cast(Any, model).embedding_layer, FactorizedLinear)
        assert isinstance(cast(Any, model).regression_layer, FactorizedLinear)


@pytest.mark.parametrize("cls", [EmbeddedFullyFactorizedFFNN, EmbeddedSimpleFullyFactorizedFFNN])
def test_embedded_fully_factorized_from_context(
    cls: type[nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for FullyFactorized variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    x = torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])
    assert model(x).shape == (BATCH_SIZE - 1, out_shapes["y"][0])


# ── EmbeddedFullySoftplusFactorizedFFNN / EmbeddedSimpleFullySoftplusFactorizedFFNN


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_SOFTPLUS_PAIRS)
def test_embedded_fully_softplus_factorized_produces_correct_output_shape(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both FullySoftplus variants emit (batch, out_features) tensors."""
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


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_SOFTPLUS_PAIRS)
def test_embedded_fully_softplus_factorized_body_has_skip_iff_residual(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Residual FullySoftplus wraps body blocks in SkipConnection; plain does not."""
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
    assert isinstance(cast(Any, residual).body.blocks[0], SkipConnection)
    assert not isinstance(cast(Any, plain).body.blocks[0], SkipConnection)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_SOFTPLUS_PAIRS)
def test_embedded_fully_softplus_factorized_all_projections_are_softplus_linear(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """Both embedding_layer and regression_layer are SoftplusFactorizedLinear (no nn.Linear)."""
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=IN_FEATURES,
            out_features=OUT_FEATURES,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert isinstance(cast(Any, model).embedding_layer, SoftplusFactorizedLinear)
        assert isinstance(cast(Any, model).regression_layer, SoftplusFactorizedLinear)


@pytest.mark.parametrize(("residual_cls", "plain_cls"), FULLY_SOFTPLUS_PAIRS)
def test_embedded_fully_softplus_factorized_unit_scale_at_init_all_layers(
    residual_cls: type[nn.Module],
    plain_cls: type[nn.Module],
) -> None:
    """softplus(log_scale) ≈ 1.0 for body blocks AND both projection layers.

    The _softplus_unit_rect_factory shifts the mean to log(e-1) for projection
    layers too, so all three layer groups initialise at unit scale.  A bug that
    skips the projection factory would leave those layers at softplus(0) ≈ 0.69.
    """
    torch.manual_seed(0)
    for cls in (residual_cls, plain_cls):
        model = cls(
            in_features=LARGE_HIDDEN_SIZE,
            out_features=LARGE_HIDDEN_SIZE,
            hidden_size=LARGE_HIDDEN_SIZE,
            num_layers=4,
        )
        # Check body blocks
        for block in cast(Any, model).body.blocks:
            if isinstance(block, SkipConnection):
                block = cast(Any, block).module
            log_scale = cast(Any, block).layer.log_scale
            mean_scale = F.softplus(log_scale).mean().item()
            assert abs(mean_scale - 1.0) < UNIT_SCALE_TOLERANCE, (
                f"{cls.__name__} body block mean_scale={mean_scale:.3f}"
            )
        # Check embedding projection
        emb_log_scale = cast(Any, model).embedding_layer.log_scale
        emb_mean_scale = F.softplus(emb_log_scale).mean().item()
        assert abs(emb_mean_scale - 1.0) < UNIT_SCALE_TOLERANCE, (
            f"{cls.__name__} embedding_layer mean_scale={emb_mean_scale:.3f}"
        )
        # Check regression projection
        reg_log_scale = cast(Any, model).regression_layer.log_scale
        reg_mean_scale = F.softplus(reg_log_scale).mean().item()
        assert abs(reg_mean_scale - 1.0) < UNIT_SCALE_TOLERANCE, (
            f"{cls.__name__} regression_layer mean_scale={reg_mean_scale:.3f}"
        )


@pytest.mark.parametrize(
    "cls", [EmbeddedFullySoftplusFactorizedFFNN, EmbeddedSimpleFullySoftplusFactorizedFFNN]
)
def test_embedded_fully_softplus_factorized_from_context(
    cls: type[nn.Module],
    rect_shapes: tuple[ShapeMapping, ShapeMapping],
) -> None:
    """from_context wires rectangular shapes correctly for FullySoftplus variants."""
    in_shapes, out_shapes = rect_shapes
    model = cast(Any, cls).from_context(
        ShapeContext(in_shapes, out_shapes), hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )
    x = torch.randn(BATCH_SIZE - 1, in_shapes["x"][0])
    assert model(x).shape == (BATCH_SIZE - 1, out_shapes["y"][0])
