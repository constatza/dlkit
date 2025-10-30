"""Tests for state dict loading in inference infrastructure.

Critical regression tests for the state_dict prefix stripping logic
that ensures trained weights are correctly loaded during inference.
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path

from dlkit.interfaces.inference.infrastructure.adapters import (
    PyTorchModelLoader,
    TorchModelStateManager,
)


class TestStateDictExtraction:
    """Test state dict extraction from Lightning checkpoints."""

    @pytest.fixture
    def model_loader(self) -> PyTorchModelLoader:
        """Create PyTorchModelLoader instance."""
        state_manager = TorchModelStateManager()
        return PyTorchModelLoader(state_manager)

    def test_extract_state_dict_with_model_prefix(self, model_loader: PyTorchModelLoader):
        """Test stripping 'model.' prefix from state dict keys."""
        # Simulate a Lightning checkpoint with model weights
        checkpoint = {
            "state_dict": {
                "model.linear.weight": torch.randn(5, 10),
                "model.linear.bias": torch.randn(5),
                "model.fc.weight": torch.randn(3, 5),
            }
        }

        result = model_loader._extract_state_dict(checkpoint)

        # Prefix should be stripped
        assert "linear.weight" in result
        assert "linear.bias" in result
        assert "fc.weight" in result
        # Original keys should not exist
        assert "model.linear.weight" not in result
        assert "model.linear.bias" not in result

    def test_extract_state_dict_with_mixed_prefixes(self, model_loader: PyTorchModelLoader):
        """Test handling Lightning checkpoints with multiple key prefixes.

        This is the CRITICAL test for the bug fix. Lightning checkpoints contain:
        - model.* (model weights - need prefix stripping)
        - fitted_transforms.* (transform state)
        - optimizer_states.* (optimizer state)

        The old logic used all() and failed here. The fix uses any().
        """
        # Realistic Lightning checkpoint structure
        checkpoint = {
            "state_dict": {
                # Model weights (need prefix stripping)
                "model.linear1.weight": torch.randn(64, 10),
                "model.linear1.bias": torch.randn(64),
                "model.linear2.weight": torch.randn(5, 64),
                "model.linear2.bias": torch.randn(5),
                # Transform state (keep as-is)
                "fitted_transforms.feature_x.transforms.0.mean": torch.tensor([0.5]),
                "fitted_transforms.feature_x.transforms.0.std": torch.tensor([0.2]),
                "fitted_transforms.target_y.transforms.0.min": torch.tensor([0.0]),
                "fitted_transforms.target_y.transforms.0.max": torch.tensor([1.0]),
            }
        }

        result = model_loader._extract_state_dict(checkpoint)

        # Model weights should have prefix stripped
        assert "linear1.weight" in result
        assert "linear1.bias" in result
        assert "linear2.weight" in result
        assert "linear2.bias" in result

        # Transform state should remain unchanged
        assert "fitted_transforms.feature_x.transforms.0.mean" in result
        assert "fitted_transforms.target_y.transforms.0.min" in result

        # Original model.* keys should not exist
        assert "model.linear1.weight" not in result

    def test_extract_state_dict_without_prefix(self, model_loader: PyTorchModelLoader):
        """Test state dict without 'model.' prefix (no stripping needed)."""
        checkpoint = {
            "state_dict": {
                "linear.weight": torch.randn(5, 10),
                "linear.bias": torch.randn(5),
                "fc.weight": torch.randn(3, 5),
            }
        }

        result = model_loader._extract_state_dict(checkpoint)

        # Keys should remain unchanged
        assert "linear.weight" in result
        assert "linear.bias" in result
        assert "fc.weight" in result
        # No model. prefix should be added
        assert "model.linear.weight" not in result

    def test_extract_state_dict_fallback_no_state_dict_key(
        self, model_loader: PyTorchModelLoader
    ):
        """Test fallback when checkpoint doesn't have 'state_dict' key."""
        # Some frameworks save weights directly without a state_dict wrapper
        checkpoint = {
            "model.linear.weight": torch.randn(5, 10),
            "model.linear.bias": torch.randn(5),
        }

        result = model_loader._extract_state_dict(checkpoint)

        # Should use entire checkpoint and strip prefix
        assert "linear.weight" in result
        assert "linear.bias" in result
        assert "model.linear.weight" not in result

    def test_extract_state_dict_preserves_values(self, model_loader: PyTorchModelLoader):
        """Test that tensor values are preserved during extraction."""
        weight_tensor = torch.randn(5, 10)
        bias_tensor = torch.randn(5)

        checkpoint = {
            "state_dict": {
                "model.linear.weight": weight_tensor,
                "model.linear.bias": bias_tensor,
            }
        }

        result = model_loader._extract_state_dict(checkpoint)

        # Values should be identical (not copied)
        assert torch.equal(result["linear.weight"], weight_tensor)
        assert torch.equal(result["linear.bias"], bias_tensor)

    def test_extract_state_dict_empty_checkpoint(self, model_loader: PyTorchModelLoader):
        """Test handling of empty checkpoint."""
        checkpoint = {"state_dict": {}}

        result = model_loader._extract_state_dict(checkpoint)

        assert result == {}


class TestStateDictLoadingIntegration:
    """Integration tests for full checkpoint loading workflow."""

    @pytest.fixture
    def sample_checkpoint_path(self, tmp_path: Path) -> Path:
        """Create a sample Lightning checkpoint for testing."""
        checkpoint_path = tmp_path / "model.ckpt"

        # Create a realistic Lightning checkpoint
        checkpoint = {
            "state_dict": {
                # Model weights
                "model.linear.weight": torch.randn(5, 10),
                "model.linear.bias": torch.randn(5),
                # Transform state (should be ignored by bare model loading)
                "fitted_transforms.x.mean": torch.tensor([0.0]),
                "fitted_transforms.x.std": torch.tensor([1.0]),
            },
            "epoch": 10,
            "global_step": 1000,
            "dlkit_metadata": {
                "version": "2.0",
                "model_settings": {
                    "name": "SimpleFFNN",
                    "module_path": "dlkit.core.models.nn.ffnn.simple",
                    "params": {"hidden_dim": 64},
                },
            },
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_checkpoint_loads_with_correct_weights(self, tmp_path: Path):
        """Verify that weights are correctly loaded (not random).

        This is the key integration test that would have caught the bug.
        If weights aren't loaded correctly, the model will have random values.
        """
        # INJECT CUSTOM KNOWN WEIGHTS (simulating trained weights)
        # Use distinctive values that are easy to verify
        known_weight = torch.arange(50, dtype=torch.float32).reshape(5, 10)  # [0, 1, 2, ..., 49]
        known_bias = torch.tensor([100.0, 200.0, 300.0, 400.0, 500.0])

        checkpoint_path = tmp_path / "trained_model.ckpt"
        checkpoint = {
            "state_dict": {
                # Injected trained weights with model. prefix (Lightning format)
                "model.linear.weight": known_weight,
                "model.linear.bias": known_bias,
                # Other Lightning state (should not interfere)
                "fitted_transforms.x.mean": torch.tensor([0.0]),
                "fitted_transforms.x.std": torch.tensor([1.0]),
            },
            "epoch": 10,
            "global_step": 1000,
        }
        torch.save(checkpoint, checkpoint_path)

        # Now load and verify the EXACT weights are extracted
        state_manager = TorchModelStateManager()
        loader = PyTorchModelLoader(state_manager)

        loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        extracted_state_dict = loader._extract_state_dict(loaded_checkpoint)

        # Critical assertions - weights must match EXACTLY
        assert "linear.weight" in extracted_state_dict
        assert "linear.bias" in extracted_state_dict

        # Verify EXACT values (not random, not uninitialized)
        assert torch.equal(extracted_state_dict["linear.weight"], known_weight), \
            "Loaded weights don't match injected trained weights!"
        assert torch.equal(extracted_state_dict["linear.bias"], known_bias), \
            "Loaded bias doesn't match injected trained bias!"

        # Verify the specific values for extra certainty
        assert extracted_state_dict["linear.weight"][0, 0].item() == 0.0
        assert extracted_state_dict["linear.weight"][0, 9].item() == 9.0
        assert extracted_state_dict["linear.weight"][4, 9].item() == 49.0
        assert extracted_state_dict["linear.bias"][0].item() == 100.0
        assert extracted_state_dict["linear.bias"][4].item() == 500.0

        # Verify old keys don't exist
        assert "model.linear.weight" not in extracted_state_dict
        assert "model.linear.bias" not in extracted_state_dict
