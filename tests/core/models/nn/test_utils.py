"""Tests for neural network utility functions.

Tests make_norm_layer and build_channel_schedule for normalization
and channel/timestep scheduling.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dlkit.core.models.nn.utils import build_channel_schedule, make_norm_layer


class TestMakeNormLayer:
    """Tests for normalization layer factory."""

    def test_none_returns_identity(self) -> None:
        """None should return Identity module."""
        layer = make_norm_layer(None, 4)
        assert isinstance(layer, nn.Identity)

    def test_none_string_returns_identity(self) -> None:
        """'none' string should return Identity module."""
        layer = make_norm_layer("none", 4)
        assert isinstance(layer, nn.Identity)

    def test_batch_returns_batchnorm(self) -> None:
        """'batch' should return BatchNorm1d."""
        layer = make_norm_layer("batch", 4)
        assert isinstance(layer, nn.BatchNorm1d)
        assert layer.num_features == 4

    def test_layer_1d_returns_layernorm(self) -> None:
        """'layer' without timesteps should return LayerNorm with scalar shape."""
        layer = make_norm_layer("layer", 4)
        assert isinstance(layer, nn.LayerNorm)
        assert layer.normalized_shape == (4,)

    def test_layer_2d_returns_layernorm_with_timesteps(self) -> None:
        """'layer' with timesteps should return LayerNorm with list shape."""
        layer = make_norm_layer("layer", 4, timesteps=8)
        assert isinstance(layer, nn.LayerNorm)
        assert list(layer.normalized_shape) == [4, 8]

    def test_instance_returns_instancenorm(self) -> None:
        """'instance' should return InstanceNorm1d."""
        layer = make_norm_layer("instance", 4)
        assert isinstance(layer, nn.InstanceNorm1d)
        assert layer.num_features == 4

    def test_unsupported_raises(self) -> None:
        """Unsupported normalizer name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported normalizer"):
            make_norm_layer("invalid_norm", 4)  # ty: ignore[invalid-argument-type]

    def test_identity_is_passthrough(self, batch_size: int = 3) -> None:
        """Identity layer should pass input unchanged."""
        layer = make_norm_layer(None, 2)
        x = torch.randn(batch_size, 2)
        assert torch.equal(layer(x), x)

    def test_batchnorm_has_learnable_params(self) -> None:
        """BatchNorm should have weight and bias parameters."""
        layer = make_norm_layer("batch", 4)
        assert hasattr(layer, "weight")
        assert hasattr(layer, "bias")
        assert layer.weight.requires_grad
        assert layer.bias.requires_grad


class TestBuildChannelSchedule:
    """Tests for channel/timestep schedule generation."""

    def test_same_start_end(self) -> None:
        """Same start and end should return constant schedule."""
        result = build_channel_schedule(4, 4, 3)
        assert result == [4, 4, 4]

    def test_increasing_schedule(self) -> None:
        """Increasing from 2 to 8 over 3 steps."""
        result = build_channel_schedule(2, 8, 3)
        assert len(result) == 3
        assert result[0] == 2
        assert result[-1] == 8
        # Should be monotonically increasing
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))

    def test_decreasing_schedule(self) -> None:
        """Decreasing from 8 to 2 over 3 steps."""
        result = build_channel_schedule(8, 2, 3)
        assert len(result) == 3
        assert result[0] == 8
        assert result[-1] == 2
        # Should be monotonically decreasing
        assert all(result[i] >= result[i + 1] for i in range(len(result) - 1))

    def test_length_matches_steps(self) -> None:
        """Result length should match requested steps."""
        for n in [2, 3, 5, 10]:
            result = build_channel_schedule(1, 10, n)
            assert len(result) == n

    def test_returns_list_of_ints(self) -> None:
        """Should return list of integers, not floats."""
        result = build_channel_schedule(2, 8, 4)
        assert isinstance(result, list)
        assert all(isinstance(v, int) for v in result)

    def test_single_step(self) -> None:
        """Single step should return list with start value."""
        result = build_channel_schedule(5, 10, 1)
        assert len(result) == 1
        assert result[0] == 5

    def test_two_steps(self) -> None:
        """Two steps should return [start, end]."""
        result = build_channel_schedule(2, 8, 2)
        assert result == [2, 8]

    def test_linear_spacing_increasing(self) -> None:
        """Verify linear spacing for increasing schedule."""
        result = build_channel_schedule(0, 10, 11)
        assert result[0] == 0
        assert result[10] == 10
        # Check approximate linear spacing (allowing for rounding)
        expected_diff = 10 / 10  # 1 per step
        for i in range(len(result) - 1):
            diff = result[i + 1] - result[i]
            assert abs(diff - expected_diff) <= 1  # Allow 1 for rounding
