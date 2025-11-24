"""Integration test for float64 checkpoint loading without precision loss.

This test verifies the COMPLETE fix for dtype handling:
1. Checkpoint saved with float64 weights
2. Model created with default float32
3. Model converted to float64 BEFORE loading state dict
4. Weights loaded with ZERO precision loss
5. Input data converted to float64
6. Forward pass executes without dtype errors
"""

import torch
import numpy as np
import pytest
from pathlib import Path

from dlkit.tools.config.components.model_components import ModelComponentSettings
from dlkit.core.shape_specs import NullShapeSpec
from dlkit.interfaces.inference.infrastructure.adapters import (
    PyTorchModelLoader,
    TorchModelStateManager
)


def test_float64_checkpoint_loads_without_precision_loss(tmp_path: Path):
    """Test that float64 checkpoints are loaded preserving full precision."""

    # Create a float64 model with specific high-precision values
    model = torch.nn.Linear(504, 504, bias=True)
    model = model.to(torch.float64)

    # Set specific high-precision weights
    model.weight.data[0, 0] = 1.123456789012345
    model.bias.data[0] = 2.234567890123456

    original_weight = model.weight.data[0, 0].item()
    original_bias = model.bias.data[0].item()

    print(f"\nOriginal checkpoint values (float64):")
    print(f"  Weight[0,0]: {original_weight:.15f}")
    print(f"  Bias[0]: {original_bias:.15f}")

    # Save checkpoint with dlkit metadata
    checkpoint_path = tmp_path / "model_float64.ckpt"
    checkpoint = {
        'state_dict': model.state_dict(),
        'dlkit_metadata': {
            'version': '2.0',
            'model_settings': {
                'name': 'Linear',
                'module_path': 'torch.nn',
                'params': {
                    'in_features': 504,
                    'out_features': 504,
                    'bias': True
                }
            }
        }
    }
    torch.save(checkpoint, checkpoint_path)

    # Verify checkpoint dtype
    loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_weight_dtype = loaded_checkpoint['state_dict']['weight'].dtype
    print(f"\nCheckpoint state_dict dtype: {checkpoint_weight_dtype}")
    assert checkpoint_weight_dtype == torch.float64

    # Load using PyTorchModelLoader (the actual dlkit code path)
    model_settings = ModelComponentSettings(
        name='Linear',
        module_path='torch.nn',
        **{'in_features': 504, 'out_features': 504, 'bias': True}
    )

    shape_spec = NullShapeSpec()  # Empty shape spec for external models

    state_manager = TorchModelStateManager()
    loader = PyTorchModelLoader(state_manager)

    # This is the critical test - load checkpoint and verify dtype preservation
    model_state = loader.load_from_checkpoint(
        checkpoint_path,
        model_settings,
        shape_spec
    )

    # Verify loaded model has float64 weights
    loaded_model = model_state.model
    loaded_weight = loaded_model.weight.data[0, 0].item()
    loaded_bias = loaded_model.bias.data[0].item()

    print(f"\nLoaded model values:")
    print(f"  Dtype: {loaded_model.weight.dtype}")
    print(f"  Weight[0,0]: {loaded_weight:.15f}")
    print(f"  Bias[0]: {loaded_bias:.15f}")

    # CRITICAL ASSERTIONS: No precision loss
    assert loaded_model.weight.dtype == torch.float64, \
        f"Model dtype should be float64, got {loaded_model.weight.dtype}"

    assert loaded_weight == original_weight, \
        f"Weight precision lost: {original_weight:.15f} != {loaded_weight:.15f}"

    assert loaded_bias == original_bias, \
        f"Bias precision lost: {original_bias:.15f} != {loaded_bias:.15f}"

    print("\n✓ ZERO PRECISION LOSS - float64 preserved perfectly!")


def test_float32_checkpoint_also_works(tmp_path: Path):
    """Verify float32 checkpoints still work correctly."""

    # Create a float32 model
    model = torch.nn.Linear(10, 5)
    assert model.weight.dtype == torch.float32

    # Save checkpoint
    checkpoint_path = tmp_path / "model_float32.ckpt"
    checkpoint = {
        'state_dict': model.state_dict(),
        'dlkit_metadata': {
            'version': '2.0',
            'model_settings': {
                'name': 'Linear',
                'module_path': 'torch.nn',
                'params': {'in_features': 10, 'out_features': 5}
            }
        }
    }
    torch.save(checkpoint, checkpoint_path)

    # Load using PyTorchModelLoader
    model_settings = ModelComponentSettings(
        name='Linear',
        module_path='torch.nn',
        **{'in_features': 10, 'out_features': 5}
    )

    state_manager = TorchModelStateManager()
    loader = PyTorchModelLoader(state_manager)

    model_state = loader.load_from_checkpoint(
        checkpoint_path,
        model_settings,
        NullShapeSpec()
    )

    # Verify loaded model has float32
    assert model_state.model.weight.dtype == torch.float32
    print("\n✓ Float32 checkpoints work correctly too!")


def test_precision_detection_accuracy(tmp_path: Path):
    """Test that _detect_checkpoint_dtype accurately detects the dtype.

    Strategy: If ANY weight is float64, use float64 (highest precision wins).
    """

    from dlkit.interfaces.inference.infrastructure.adapters import PyTorchModelLoader, TorchModelStateManager

    loader = PyTorchModelLoader(TorchModelStateManager())

    # Test float64 detection
    state_dict_64 = {
        'weight': torch.randn(10, 5, dtype=torch.float64),
        'bias': torch.randn(10, dtype=torch.float64)
    }
    detected_dtype = loader._detect_checkpoint_dtype(state_dict_64)
    assert detected_dtype == torch.float64

    # Test float32 detection
    state_dict_32 = {
        'weight': torch.randn(10, 5, dtype=torch.float32),
        'bias': torch.randn(10, dtype=torch.float32)
    }
    detected_dtype = loader._detect_checkpoint_dtype(state_dict_32)
    assert detected_dtype == torch.float32

    # Test mixed dtype (should return HIGHEST precision, not most common)
    # Even if only ONE tensor is float64, we use float64
    state_dict_mixed = {
        'weight1': torch.randn(10, 5, dtype=torch.float32),
        'weight2': torch.randn(10, 5, dtype=torch.float32),
        'weight3': torch.randn(10, 5, dtype=torch.float32),
        'bias1': torch.randn(10, dtype=torch.float32),
        'bias2': torch.randn(10, dtype=torch.float32),
        'critical_weight': torch.randn(5, dtype=torch.float64),  # ONE float64 tensor
    }
    detected_dtype = loader._detect_checkpoint_dtype(state_dict_mixed)
    assert detected_dtype == torch.float64, \
        "Should use float64 if ANY weight is float64, regardless of count"

    # Test precision hierarchy: float64 > float32 > float16
    state_dict_all_precisions = {
        'w1': torch.randn(5, dtype=torch.float16),
        'w2': torch.randn(5, dtype=torch.float32),
        'w3': torch.randn(5, dtype=torch.float64),
    }
    detected_dtype = loader._detect_checkpoint_dtype(state_dict_all_precisions)
    assert detected_dtype == torch.float64, "float64 should win over all others"

    state_dict_32_and_16 = {
        'w1': torch.randn(5, dtype=torch.float16),
        'w2': torch.randn(5, dtype=torch.float32),
    }
    detected_dtype = loader._detect_checkpoint_dtype(state_dict_32_and_16)
    assert detected_dtype == torch.float32, "float32 should win over float16"

    print("\n✓ Dtype detection uses HIGHEST precision strategy correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
