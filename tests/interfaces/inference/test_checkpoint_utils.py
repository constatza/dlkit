"""Tests for checkpoint loading utilities.

Tests the consolidated checkpoint loading functions from the simplified
inference subsystem architecture.
"""

from __future__ import annotations

import pytest
import torch

from dlkit.interfaces.inference.loading import extract_state_dict


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
