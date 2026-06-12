"""Tests for SkipConnection, skip layer factory functions, and ResidualSequential."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from dlkit.domain.nn.primitives.skip import (
    ResidualSequential,
    SkipConnection,
    build_conv1d_skip_layer,
    build_linear_skip_layer,
)


def test_skip_sum_output_channels(
    linear_skip_sum: SkipConnection, skip_input: torch.Tensor
) -> None:
    """Sum mode preserves out_channels."""
    assert linear_skip_sum(skip_input).shape == (3, 8)


def test_skip_concat_output_channels(
    linear_skip_concat: SkipConnection, skip_input: torch.Tensor
) -> None:
    """Concat mode produces 2 x out_channels."""
    out = linear_skip_concat(skip_input)
    assert out.shape == (3, 16), f"concat doubles channels: expected (3,16), got {out.shape}"


def test_skip_no_activation_param() -> None:
    """activation parameter must not exist on SkipConnection."""
    sig = inspect.signature(SkipConnection.__init__)
    assert "activation" not in sig.parameters


def test_no_layer_type_param() -> None:
    """layer_type parameter must not exist on SkipConnection."""
    sig = inspect.signature(SkipConnection.__init__)
    assert "layer_type" not in sig.parameters


def test_no_kernel_size_param() -> None:
    """kernel_size parameter must not exist on SkipConnection."""
    sig = inspect.signature(SkipConnection.__init__)
    assert "kernel_size" not in sig.parameters


def test_invalid_how_raises() -> None:
    """Unknown aggregation string raises ValueError."""
    module = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="Unknown aggregation"):
        SkipConnection(module, build_linear_skip_layer(module), how=cast("Any", "multiply"))


def test_bias_linear_skip_respected_true() -> None:
    """build_linear_skip_layer with bias=True produces a Linear with bias."""
    module = nn.Linear(4, 8)
    adapter = build_linear_skip_layer(module, bias=True)
    assert isinstance(adapter, nn.Linear)
    assert adapter.bias is not None


def test_bias_linear_skip_respected_false() -> None:
    """build_linear_skip_layer with bias=False produces a Linear without bias."""
    module = nn.Linear(4, 8)
    adapter = build_linear_skip_layer(module, bias=False)
    assert isinstance(adapter, nn.Linear)
    assert adapter.bias is None


def test_same_channels_linear_returns_identity() -> None:
    """build_linear_skip_layer returns Identity when in==out."""
    module = nn.Linear(8, 8)
    assert isinstance(build_linear_skip_layer(module), nn.Identity)


def test_stride_identity_not_returned(
    conv_skip_stride2: SkipConnection, basic_input: torch.Tensor
) -> None:
    """Conv skip with stride=2 does NOT use Identity; forward succeeds."""
    assert not isinstance(conv_skip_stride2.reduce_layer, nn.Identity)
    out = conv_skip_stride2(basic_input)
    assert out.shape[0] == basic_input.shape[0]
    assert out.shape[1] == basic_input.shape[1]


def test_effective_out_channels_sum(linear_skip_sum: SkipConnection) -> None:
    """effective_out_channels equals out_channels in sum mode."""
    assert linear_skip_sum.effective_out_channels == 8


def test_effective_out_channels_concat(linear_skip_concat: SkipConnection) -> None:
    """effective_out_channels equals 2 x out_channels in concat mode."""
    assert linear_skip_concat.effective_out_channels == 16


def test_build_conv1d_same_channels_stride1_returns_identity() -> None:
    """build_conv1d_skip_layer with same channels and stride=1 returns Identity."""
    module = nn.Conv1d(8, 8, 3)
    assert isinstance(build_conv1d_skip_layer(module), nn.Identity)


def test_build_conv1d_different_channels_returns_conv() -> None:
    """build_conv1d_skip_layer with different channels returns Conv1d."""
    module = nn.Conv1d(4, 8, 3)
    adapter = build_conv1d_skip_layer(module)
    assert isinstance(adapter, nn.Conv1d)


# ── ResidualSequential tests ──────────────────────────────────────────────────

_RS_DIM = 8
_RS_SEED = 77


@pytest.fixture
def rs_input() -> Tensor:
    """Batch of 4 samples with _RS_DIM features for ResidualSequential tests."""
    torch.manual_seed(_RS_SEED)
    return torch.randn(4, _RS_DIM)


@pytest.fixture
def rs_linear_chain() -> list[nn.Linear]:
    """Two-module chain: Linear(_RS_DIM, _RS_DIM) × 2."""
    torch.manual_seed(_RS_SEED + 1)
    return [nn.Linear(_RS_DIM, _RS_DIM), nn.Linear(_RS_DIM, _RS_DIM)]


@pytest.fixture
def rs_no_shortcut(rs_linear_chain: list[nn.Linear]) -> ResidualSequential:
    """ResidualSequential with no explicit shortcut (identity skip)."""
    return ResidualSequential(*rs_linear_chain)


@pytest.fixture
def rs_shortcut() -> nn.Linear:
    """Linear projection shortcut for ResidualSequential."""
    torch.manual_seed(_RS_SEED + 2)
    return nn.Linear(_RS_DIM, _RS_DIM)


@pytest.fixture
def rs_with_shortcut(
    rs_linear_chain: list[nn.Linear], rs_shortcut: nn.Linear
) -> ResidualSequential:
    """ResidualSequential with an explicit linear shortcut."""
    return ResidualSequential(*rs_linear_chain, shortcut=rs_shortcut)


def test_residual_sequential_no_shortcut_equals_chain_plus_identity(
    rs_no_shortcut: ResidualSequential,
    rs_linear_chain: list[nn.Linear],
    rs_input: Tensor,
) -> None:
    """Without shortcut: output = chain(x) + x."""
    with torch.no_grad():
        chain_out = rs_linear_chain[1](rs_linear_chain[0](rs_input))
        expected = chain_out + rs_input
        actual = rs_no_shortcut(rs_input)
    assert torch.allclose(actual, expected)


def test_residual_sequential_with_shortcut_equals_chain_plus_shortcut(
    rs_with_shortcut: ResidualSequential,
    rs_linear_chain: list[nn.Linear],
    rs_shortcut: nn.Linear,
    rs_input: Tensor,
) -> None:
    """With explicit shortcut: output = chain(x) + shortcut(x)."""
    with torch.no_grad():
        chain_out = rs_linear_chain[1](rs_linear_chain[0](rs_input))
        expected = chain_out + rs_shortcut(rs_input)
        actual = rs_with_shortcut(rs_input)
    assert torch.allclose(actual, expected)


def test_residual_sequential_no_shortcut_attribute_is_none(
    rs_no_shortcut: ResidualSequential,
) -> None:
    """shortcut attribute is None when none is provided."""
    assert rs_no_shortcut.shortcut is None


def test_residual_sequential_shortcut_attribute_is_provided_module(
    rs_with_shortcut: ResidualSequential, rs_shortcut: nn.Linear
) -> None:
    """shortcut attribute is the provided nn.Linear."""
    assert rs_with_shortcut.shortcut is rs_shortcut


def test_residual_sequential_parameters_contains_all_chain_params(
    rs_no_shortcut: ResidualSequential, rs_linear_chain: list[nn.Linear]
) -> None:
    """All chain module parameters appear in parameters()."""
    all_param_ids = {id(p) for p in rs_no_shortcut.parameters()}
    for module in rs_linear_chain:
        for p in module.parameters():
            assert id(p) in all_param_ids


def test_residual_sequential_parameters_contains_shortcut_params(
    rs_with_shortcut: ResidualSequential,
    rs_shortcut: nn.Linear,
) -> None:
    """Shortcut parameters appear in parameters() when shortcut is provided."""
    all_param_ids = {id(p) for p in rs_with_shortcut.parameters()}
    for p in rs_shortcut.parameters():
        assert id(p) in all_param_ids


@pytest.fixture
def single_layer() -> nn.Linear:
    """Single nn.Linear(_RS_DIM, _RS_DIM) for minimal chain tests."""
    torch.manual_seed(_RS_SEED + 3)
    return nn.Linear(_RS_DIM, _RS_DIM)


@pytest.fixture
def single_module_residual_sequential(single_layer: nn.Linear) -> ResidualSequential:
    """Single-layer ResidualSequential wrapping single_layer."""
    return ResidualSequential(single_layer)


def test_residual_sequential_single_module_chain(
    rs_input: Tensor,
    single_layer: nn.Linear,
    single_module_residual_sequential: ResidualSequential,
) -> None:
    """Single-module chain: output = module(x) + x."""
    with torch.no_grad():
        expected = single_layer(rs_input) + rs_input
        actual = single_module_residual_sequential(rs_input)
    assert torch.allclose(actual, expected)


def test_residual_sequential_multi_module_chain_output_shape(
    rs_no_shortcut: ResidualSequential, rs_input: Tensor
) -> None:
    """Multi-module chain output shape matches input shape."""
    with torch.no_grad():
        out = rs_no_shortcut(rs_input)
    assert out.shape == rs_input.shape
