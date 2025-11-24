"""Tests for shared checkpoint utilities.

Tests the DRY refactored checkpoint loading utilities used across
all inference infrastructure components.
"""

from __future__ import annotations

import pytest
import torch

from dlkit.interfaces.inference.checkpoint_utils import (
    extract_state_dict,
    _has_model_prefix,
    _strip_prefix_if_present,
    _strip_model_prefix,
    _extract_raw_state_dict,
)


class TestPureFunctions:
    """Test pure helper functions."""

    def test_has_model_prefix_with_prefix(self):
        """Test detection of model. prefix."""
        keys = ["model.weight", "model.bias", "other.param"]
        assert _has_model_prefix(keys) is True

    def test_has_model_prefix_without_prefix(self):
        """Test detection when no model. prefix."""
        keys = ["weight", "bias", "param"]
        assert _has_model_prefix(keys) is False

    def test_has_model_prefix_empty(self):
        """Test detection with empty list."""
        assert _has_model_prefix([]) is False

    def test_strip_prefix_if_present_with_prefix(self):
        """Test stripping prefix when present."""
        assert _strip_prefix_if_present("model.weight") == "weight"
        assert _strip_prefix_if_present("model.linear.bias") == "linear.bias"

    def test_strip_prefix_if_present_without_prefix(self):
        """Test no stripping when prefix absent."""
        assert _strip_prefix_if_present("weight") == "weight"
        assert _strip_prefix_if_present("fitted_transforms.x.mean") == "fitted_transforms.x.mean"

    def test_strip_model_prefix(self):
        """Test batch prefix stripping."""
        state_dict = {
            "model.weight": torch.tensor([1.0]),
            "model.bias": torch.tensor([2.0]),
            "fitted_transforms.mean": torch.tensor([0.0]),
        }

        result = _strip_model_prefix(state_dict)

        assert "weight" in result
        assert "bias" in result
        assert "fitted_transforms.mean" in result
        assert "model.weight" not in result

    def test_extract_raw_state_dict_lightning_format(self):
        """Test extraction from Lightning format."""
        checkpoint = {"state_dict": {"weight": torch.tensor([1.0])}}
        result = _extract_raw_state_dict(checkpoint)
        assert "weight" in result

    def test_extract_raw_state_dict_pytorch_format(self):
        """Test extraction from PyTorch format."""
        checkpoint = {"model_state_dict": {"weight": torch.tensor([1.0])}}
        result = _extract_raw_state_dict(checkpoint)
        assert "weight" in result

    def test_extract_raw_state_dict_fallback(self):
        """Test fallback when no wrapper key."""
        checkpoint = {"weight": torch.tensor([1.0])}
        result = _extract_raw_state_dict(checkpoint)
        assert result == checkpoint


class TestExtractStateDict:
    """Test main extraction function composition."""

    def test_extract_with_model_prefix(self):
        """Test extraction strips model. prefix."""
        checkpoint = {
            "state_dict": {
                "model.linear.weight": torch.randn(5, 10),
                "model.linear.bias": torch.randn(5),
            }
        }

        result = extract_state_dict(checkpoint)

        assert "linear.weight" in result
        assert "linear.bias" in result
        assert "model.linear.weight" not in result

    def test_extract_with_mixed_prefixes(self):
        """Test extraction handles mixed prefixes correctly."""
        checkpoint = {
            "state_dict": {
                "model.weight": torch.randn(5, 10),
                "fitted_transforms.mean": torch.tensor([0.0]),
                "optimizer_states.lr": torch.tensor([0.001]),
            }
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result
        assert "fitted_transforms.mean" in result
        assert "optimizer_states.lr" in result
        assert "model.weight" not in result

    def test_extract_without_prefix(self):
        """Test extraction preserves keys without prefix."""
        checkpoint = {
            "state_dict": {
                "weight": torch.randn(5, 10),
                "bias": torch.randn(5),
            }
        }

        result = extract_state_dict(checkpoint)

        assert "weight" in result
        assert "bias" in result

    def test_extract_preserves_values(self):
        """Test tensor values are preserved."""
        weight = torch.randn(5, 10)
        checkpoint = {"state_dict": {"model.weight": weight}}

        result = extract_state_dict(checkpoint)

        assert torch.equal(result["weight"], weight)

    def test_extract_empty_checkpoint(self):
        """Test handling empty checkpoint."""
        result = extract_state_dict({"state_dict": {}})
        assert result == {}

    def test_extract_non_dict_checkpoint(self):
        """Test handling non-dict checkpoint."""
        checkpoint = "not_a_dict"
        result = extract_state_dict(checkpoint)
        assert result == "not_a_dict"
