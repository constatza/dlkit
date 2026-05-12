"""Tests for SkipConnection."""

from __future__ import annotations

import inspect

import torch

from dlkit.domain.nn.primitives.skip import SkipConnection


def test_skip_sum_output_channels(
    linear_skip_sum: SkipConnection, skip_input: torch.Tensor
) -> None:
    """Sum mode preserves out_channels."""
    assert linear_skip_sum(skip_input).shape == (3, 8)


def test_skip_concat_output_channels(
    linear_skip_concat: SkipConnection, skip_input: torch.Tensor
) -> None:
    """concat mode produces 2×out_channels."""
    out = linear_skip_concat(skip_input)
    assert out.shape == (3, 16), f"concat doubles channels: expected (3,16), got {out.shape}"


def test_skip_no_activation_param() -> None:
    """activation parameter must not exist on SkipConnection."""
    sig = inspect.signature(SkipConnection.__init__)
    assert "activation" not in sig.parameters
