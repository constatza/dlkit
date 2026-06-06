"""Tests for SkipConnection and skip layer factory functions."""

from __future__ import annotations

import inspect
from typing import Any, cast

import pytest
import torch
from torch import nn

from dlkit.domain.nn.primitives.skip import (
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
