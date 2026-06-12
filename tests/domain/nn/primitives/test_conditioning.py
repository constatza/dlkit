"""Tests for conditioned neural network primitives.

Covers IConditionedModule, AsConditioned, FiLMLayer,
ConditionedSequential, and ConditionedResidualSequential.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from dlkit.domain.nn.primitives.conditioning import (
    AsConditioned,
    ConditionedResidualSequential,
    ConditionedSequential,
    FiLMLayer,
    IConditionedModule,
)

# ── Constants ──────────────────────────────────────────────────────────────────

BATCH = 4
FEATURE_DIM = 8
CONDITION_DIM = 6
SEED = 42
IDENTITY_ATOL = 1e-6


# ── Minimal concrete subclass ──────────────────────────────────────────────────


class _DoubleBlock(IConditionedModule):
    """Doubles x; condition is used to add a linear projection for variety."""

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(condition_dim, feature_dim)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        return 2.0 * x + self.proj(condition)


class _PassthroughBlock(IConditionedModule):
    """Returns x unchanged; condition is ignored for order-sensitive tests."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        return self.linear(x)


class _ConditionSensitive(IConditionedModule):
    """Adds mean of condition to x (shapes always compatible)."""

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        return x + condition.mean(dim=-1, keepdim=True)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def feature_input() -> Tensor:
    """Batched feature tensor of shape (BATCH, FEATURE_DIM)."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH, FEATURE_DIM)


@pytest.fixture
def condition_input() -> Tensor:
    """Batched condition tensor of shape (BATCH, CONDITION_DIM)."""
    torch.manual_seed(SEED + 1)
    return torch.randn(BATCH, CONDITION_DIM)


@pytest.fixture
def inner_linear() -> nn.Linear:
    """Plain nn.Linear to wrap with AsConditioned."""
    torch.manual_seed(SEED)
    return nn.Linear(FEATURE_DIM, FEATURE_DIM)


@pytest.fixture
def as_conditioned(inner_linear: nn.Linear) -> AsConditioned:
    """AsConditioned wrapping a plain linear."""
    return AsConditioned(inner_linear)


@pytest.fixture
def film_layer() -> FiLMLayer:
    """Zero-initialised FiLMLayer(CONDITION_DIM -> FEATURE_DIM)."""
    return FiLMLayer(condition_dim=CONDITION_DIM, feature_dim=FEATURE_DIM)


@pytest.fixture
def double_block() -> _DoubleBlock:
    """Concrete IConditionedModule that doubles the input."""
    torch.manual_seed(SEED)
    return _DoubleBlock(feature_dim=FEATURE_DIM, condition_dim=CONDITION_DIM)


@pytest.fixture
def passthrough_block() -> _PassthroughBlock:
    """Concrete IConditionedModule that applies a linear to x."""
    torch.manual_seed(SEED)
    return _PassthroughBlock(FEATURE_DIM)


@pytest.fixture
def conditioned_sequential(
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
) -> ConditionedSequential:
    """ConditionedSequential with two blocks."""
    return ConditionedSequential(double_block, passthrough_block)


@pytest.fixture
def conditioned_residual_no_shortcut(
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
) -> ConditionedResidualSequential:
    """ConditionedResidualSequential without explicit shortcut (identity skip)."""
    return ConditionedResidualSequential(double_block, passthrough_block)


@pytest.fixture
def shortcut_linear() -> nn.Linear:
    """Linear shortcut for ConditionedResidualSequential."""
    torch.manual_seed(SEED + 2)
    return nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=True)


@pytest.fixture
def conditioned_residual_with_shortcut(
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
    shortcut_linear: nn.Linear,
) -> ConditionedResidualSequential:
    """ConditionedResidualSequential with an explicit linear shortcut."""
    return ConditionedResidualSequential(double_block, passthrough_block, shortcut=shortcut_linear)


@pytest.fixture
def gamma_beta_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """Feature tensor x and condition tensor c for FiLM formula verification.

    Returns:
        Tuple of (x, c) where x has shape (BATCH, 4) and c has shape (BATCH, 3).
    """
    torch.manual_seed(SEED)
    x = torch.randn(BATCH, 4)
    torch.manual_seed(SEED + 1)
    c = torch.randn(BATCH, 3)
    return x, c


@pytest.fixture
def film_layer_gamma_one_beta_two(
    gamma_beta_tensors: tuple[torch.Tensor, torch.Tensor],
) -> FiLMLayer:
    """FiLMLayer with gamma bias=1 and beta bias=2 for formula verification.

    Dimensions are derived from the ``gamma_beta_tensors`` fixture so that
    condition_dim and feature_dim are always consistent.

    Args:
        gamma_beta_tensors: Provides the (x, c) tensors whose shapes determine
            the layer dimensions.

    Returns:
        A FiLMLayer with to_gamma.weight=0, to_gamma.bias=1,
        to_beta.weight=0, to_beta.bias=2.
    """
    x, c = gamma_beta_tensors
    layer = FiLMLayer(condition_dim=c.shape[-1], feature_dim=x.shape[-1])
    with torch.no_grad():
        nn.init.zeros_(layer.to_gamma.weight)
        nn.init.zeros_(layer.to_gamma.bias)
        layer.to_gamma.bias.fill_(1.0)
        nn.init.zeros_(layer.to_beta.weight)
        layer.to_beta.bias.fill_(2.0)
    return layer


@pytest.fixture
def conditioned_sequential_reversed(
    passthrough_block: _PassthroughBlock,
    double_block: _DoubleBlock,
) -> ConditionedSequential:
    """ConditionedSequential with blocks in reverse order: passthrough then double."""
    return ConditionedSequential(passthrough_block, double_block)


@pytest.fixture
def large_batch_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Large-batch feature and condition tensors for batch-size variability tests.

    Returns:
        Tuple of (x, c) where x has shape (16, FEATURE_DIM) and c has shape (16, CONDITION_DIM).
    """
    torch.manual_seed(SEED + 4)
    x = torch.randn(16, FEATURE_DIM)
    torch.manual_seed(SEED + 5)
    c = torch.randn(16, CONDITION_DIM)
    return x, c


@pytest.fixture
def condition_sensitive_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Feature tensor x and an alternate condition for condition-reach tests.

    Returns:
        Tuple of (x, other_condition) both with shape (BATCH, CONDITION_DIM).
    """
    torch.manual_seed(SEED + 6)
    x = torch.randn(BATCH, CONDITION_DIM)
    torch.manual_seed(SEED + 7)
    other_condition = torch.randn(BATCH, CONDITION_DIM)
    return x, other_condition


@pytest.fixture
def other_condition_input() -> Tensor:
    """Alternate batched condition tensor of shape (BATCH, CONDITION_DIM).

    Uses a different seed than ``condition_input`` so the tensors are distinct.
    """
    torch.manual_seed(SEED + 3)
    return torch.randn(BATCH, CONDITION_DIM)


@pytest.fixture
def condition_sensitive_sequential() -> ConditionedSequential:
    """ConditionedSequential composed of two _ConditionSensitive blocks.

    Returns:
        A ConditionedSequential where every block responds to the condition.
    """
    block1 = _ConditionSensitive()
    block2 = _ConditionSensitive()
    return ConditionedSequential(block1, block2)


@pytest.fixture
def conditioned_body_output(
    conditioned_sequential: ConditionedSequential,
    feature_input: Tensor,
    condition_input: Tensor,
) -> Tensor:
    """Output of running ConditionedSequential(double_block, passthrough_block) on inputs.

    Args:
        conditioned_sequential: The sequential body composed of double_block and passthrough_block.
        feature_input: Batched feature tensor of shape (BATCH, FEATURE_DIM).
        condition_input: Batched condition tensor of shape (BATCH, CONDITION_DIM).

    Returns:
        The body output tensor produced with no gradient tracking.
    """
    with torch.no_grad():
        return conditioned_sequential(feature_input, condition_input)


# ── IConditionedModule tests ───────────────────────────────────────────────────


def test_iconditioned_module_cannot_be_instantiated_directly() -> None:
    """Direct instantiation of IConditionedModule must raise TypeError."""
    with pytest.raises(TypeError):
        IConditionedModule()  # type: ignore[abstract]


def test_iconditioned_module_concrete_subclass_can_be_instantiated(
    double_block: _DoubleBlock,
) -> None:
    """A concrete subclass with forward(x, condition) can be constructed."""
    assert isinstance(double_block, IConditionedModule)


def test_iconditioned_module_is_nn_module_subclass(double_block: _DoubleBlock) -> None:
    """Concrete IConditionedModule instances are also nn.Module instances."""
    assert isinstance(double_block, nn.Module)


def test_iconditioned_module_concrete_subclass_is_abstract_subtype() -> None:
    """Subclass without forward still raises TypeError at instantiation."""

    class _NoForward(IConditionedModule):
        pass

    with pytest.raises(TypeError):
        _NoForward()  # type: ignore[abstract]


# ── AsConditioned tests ────────────────────────────────────────────────────────


def test_as_conditioned_output_equals_wrapped_module_output(
    as_conditioned: AsConditioned,
    inner_linear: nn.Linear,
    feature_input: Tensor,
    condition_input: Tensor,
) -> None:
    """forward(x, condition) must equal wrapped_module(x)."""
    expected = inner_linear(feature_input)
    actual = as_conditioned(feature_input, condition_input)
    assert torch.allclose(actual, expected)


def test_as_conditioned_condition_is_ignored(
    as_conditioned: AsConditioned,
    feature_input: Tensor,
    condition_input: Tensor,
    other_condition_input: Tensor,
) -> None:
    """Different condition tensors must produce the same output."""
    out1 = as_conditioned(feature_input, condition_input)
    out2 = as_conditioned(feature_input, other_condition_input)
    assert torch.allclose(out1, out2)


def test_as_conditioned_tracks_wrapped_module_parameters(
    as_conditioned: AsConditioned,
    inner_linear: nn.Linear,
) -> None:
    """Parameter count of AsConditioned must equal that of the wrapped module."""
    wrapped_params = set(inner_linear.parameters())
    container_params = set(as_conditioned.parameters())
    assert wrapped_params == container_params


def test_as_conditioned_is_iconditioned_module(as_conditioned: AsConditioned) -> None:
    """AsConditioned must satisfy the IConditionedModule protocol."""
    assert isinstance(as_conditioned, IConditionedModule)


# ── FiLMLayer tests ────────────────────────────────────────────────────────────


def test_film_layer_identity_at_init(
    film_layer: FiLMLayer, feature_input: Tensor, condition_input: Tensor
) -> None:
    """With zero-init weights, FiLMLayer(x, c) == x for any input."""
    with torch.no_grad():
        out = film_layer(feature_input, condition_input)
    assert torch.allclose(out, feature_input, atol=IDENTITY_ATOL)


def test_film_layer_output_shape_matches_feature_dim(
    film_layer: FiLMLayer, feature_input: Tensor, condition_input: Tensor
) -> None:
    """Output shape must match (BATCH, FEATURE_DIM)."""
    with torch.no_grad():
        out = film_layer(feature_input, condition_input)
    assert out.shape == feature_input.shape


def test_film_layer_applies_gamma_beta_formula(
    gamma_beta_tensors: tuple[torch.Tensor, torch.Tensor],
    film_layer_gamma_one_beta_two: FiLMLayer,
) -> None:
    """Manually set gamma/beta weights verify (1+gamma)*x + beta formula."""
    x, c = gamma_beta_tensors
    # Expected: (1 + 1) * x + 2 = 2x + 2
    with torch.no_grad():
        out = film_layer_gamma_one_beta_two(x, c)
    expected = 2.0 * x + 2.0
    assert torch.allclose(out, expected, atol=1e-5)


def test_film_layer_works_with_batched_inputs(
    film_layer: FiLMLayer,
    large_batch_inputs: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """FiLMLayer handles a varying batch size correctly."""
    large_batch_x, large_batch_c = large_batch_inputs
    with torch.no_grad():
        out = film_layer(large_batch_x, large_batch_c)
    assert out.shape == large_batch_x.shape


# ── ConditionedSequential tests ────────────────────────────────────────────────


def test_conditioned_sequential_applies_blocks_in_order(
    feature_input: Tensor,
    condition_input: Tensor,
    conditioned_sequential: ConditionedSequential,
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
) -> None:
    """Blocks are applied in declaration order: double then passthrough."""
    with torch.no_grad():
        out = conditioned_sequential(feature_input, condition_input)

    # Manually replicate: passthrough_block(double_block(x, c), c)
    with torch.no_grad():
        after_double = double_block(feature_input, condition_input)
        expected = passthrough_block(after_double, condition_input)

    assert torch.allclose(out, expected)


def test_conditioned_sequential_order_matters(
    feature_input: Tensor,
    condition_input: Tensor,
    conditioned_sequential: ConditionedSequential,
    conditioned_sequential_reversed: ConditionedSequential,
) -> None:
    """Reversing block order changes the result (order sensitivity check)."""
    with torch.no_grad():
        out_fwd = conditioned_sequential(feature_input, condition_input)
        out_rev = conditioned_sequential_reversed(feature_input, condition_input)
    assert not torch.allclose(out_fwd, out_rev)


def test_conditioned_sequential_condition_reaches_all_blocks(
    condition_input: Tensor,
    condition_sensitive_inputs: tuple[torch.Tensor, torch.Tensor],
    condition_sensitive_sequential: ConditionedSequential,
) -> None:
    """Every block receives the same condition; perturbing it changes output."""
    # Use CONDITION_DIM-sized features so shapes are unambiguous
    x, other_condition = condition_sensitive_inputs

    with torch.no_grad():
        out1 = condition_sensitive_sequential(x, condition_input)
        out2 = condition_sensitive_sequential(x, other_condition)

    assert not torch.allclose(out1, out2)


def test_conditioned_sequential_parameters_contains_all_block_params(
    conditioned_sequential: ConditionedSequential,
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
) -> None:
    """All block parameters appear in ConditionedSequential.parameters()."""
    seq_param_ids = {id(p) for p in conditioned_sequential.parameters()}
    for block in (double_block, passthrough_block):
        for p in block.parameters():
            assert id(p) in seq_param_ids


# ── ConditionedResidualSequential tests ───────────────────────────────────────


def test_conditioned_residual_no_shortcut_identity_skip(
    feature_input: Tensor,
    condition_input: Tensor,
    conditioned_residual_no_shortcut: ConditionedResidualSequential,
    conditioned_body_output: Tensor,
) -> None:
    """Without shortcut: output = body(x, c) + x."""
    expected = conditioned_body_output + feature_input
    with torch.no_grad():
        actual = conditioned_residual_no_shortcut(feature_input, condition_input)
    assert torch.allclose(actual, expected)


def test_conditioned_residual_with_explicit_shortcut(
    feature_input: Tensor,
    condition_input: Tensor,
    conditioned_residual_with_shortcut: ConditionedResidualSequential,
    conditioned_body_output: Tensor,
    shortcut_linear: nn.Linear,
) -> None:
    """With explicit shortcut: output = body(x, c) + shortcut(x)."""
    with torch.no_grad():
        expected = conditioned_body_output + shortcut_linear(feature_input)
        actual = conditioned_residual_with_shortcut(feature_input, condition_input)
    assert torch.allclose(actual, expected)


def test_conditioned_residual_shortcut_none_uses_identity(
    conditioned_residual_no_shortcut: ConditionedResidualSequential,
) -> None:
    """shortcut attribute is None when no shortcut is provided."""
    assert conditioned_residual_no_shortcut.shortcut is None


def test_conditioned_residual_with_shortcut_not_none(
    conditioned_residual_with_shortcut: ConditionedResidualSequential,
    shortcut_linear: nn.Linear,
) -> None:
    """shortcut attribute is the provided linear when shortcut is given."""
    assert conditioned_residual_with_shortcut.shortcut is shortcut_linear


def test_conditioned_residual_parameters_contains_all_params(
    conditioned_residual_with_shortcut: ConditionedResidualSequential,
    double_block: _DoubleBlock,
    passthrough_block: _PassthroughBlock,
    shortcut_linear: nn.Linear,
) -> None:
    """All body and shortcut parameters appear in the module's parameters()."""
    all_param_ids = {id(p) for p in conditioned_residual_with_shortcut.parameters()}
    for module in (double_block, passthrough_block, shortcut_linear):
        for p in module.parameters():
            assert id(p) in all_param_ids
