"""Tests for NPZ file loading functionality in arrays.py.

Tests cover:
- Single-array NPZ auto-detection
- Multi-array NPZ with explicit keys
- Error handling for missing keys
- Precision integration with PrecisionService
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from dlkit.tools.io.arrays import _load_npz, load_array
from dlkit.tools.precision import PrecisionStrategy, precision_override


class TestLoadNpzFunction:
    """Test _load_npz helper function."""

    def test_load_single_array_auto_detect(self, npz_single_array: dict) -> None:
        """Test auto-detection for single-array NPZ files."""
        result = _load_npz(npz_single_array["path"])

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        np.testing.assert_array_equal(result, npz_single_array["array"])

    def test_load_multi_array_with_explicit_key(self, npz_multi_array: dict) -> None:
        """Test loading specific array from multi-array NPZ with explicit key."""
        # Load features
        features = _load_npz(npz_multi_array["path"], array_key="features")
        assert isinstance(features, np.ndarray)
        assert features.shape == (10, 5)
        np.testing.assert_array_equal(features, npz_multi_array["features"])

        # Load targets
        targets = _load_npz(npz_multi_array["path"], array_key="targets")
        assert isinstance(targets, np.ndarray)
        assert targets.shape == (10, 1)
        np.testing.assert_array_equal(targets, npz_multi_array["targets"])

        # Load latent
        latent = _load_npz(npz_multi_array["path"], array_key="latent")
        assert isinstance(latent, np.ndarray)
        assert latent.shape == (10, 3)
        np.testing.assert_array_equal(latent, npz_multi_array["latent"])

    def test_load_multi_array_without_key_raises(self, npz_multi_array: dict) -> None:
        """Test that multi-array NPZ without key raises clear error."""
        with pytest.raises(ValueError, match="contains multiple arrays"):
            _load_npz(npz_multi_array["path"])

    def test_load_with_missing_key_raises(self, npz_multi_array: dict) -> None:
        """Test that missing array key raises clear error."""
        with pytest.raises(ValueError, match="Array key 'missing' not found"):
            _load_npz(npz_multi_array["path"], array_key="missing")

    def test_load_empty_npz_raises(self, npz_empty: Path) -> None:
        """Test that empty NPZ file raises appropriate error."""
        with pytest.raises(ValueError, match="contains multiple arrays"):
            _load_npz(npz_empty)


class TestLoadArrayWithNpz:
    """Test load_array function with NPZ files."""

    def test_load_single_array_npz_returns_tensor(self, npz_single_array: dict) -> None:
        """Test that load_array converts single-array NPZ to tensor."""
        result = load_array(npz_single_array["path"])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (10, 5)
        assert result.dtype == torch.float32  # Default precision

    def test_load_multi_array_npz_with_key(self, npz_multi_array: dict) -> None:
        """Test load_array with explicit array_key for multi-array NPZ."""
        features = load_array(npz_multi_array["path"], array_key="features")
        assert isinstance(features, torch.Tensor)
        assert features.shape == (10, 5)

        targets = load_array(npz_multi_array["path"], array_key="targets")
        assert isinstance(targets, torch.Tensor)
        assert targets.shape == (10, 1)

    def test_load_npz_respects_precision_context(self, npz_single_array: dict) -> None:
        """Test that NPZ loading respects precision context."""
        # Test with float64 precision
        with precision_override(PrecisionStrategy.FULL_64):
            result = load_array(npz_single_array["path"])
            assert result.dtype == torch.float64

        # Test with float32 precision (default)
        with precision_override(PrecisionStrategy.FULL_32):
            result = load_array(npz_single_array["path"])
            assert result.dtype == torch.float32

    def test_load_npz_with_explicit_dtype(self, npz_single_array: dict) -> None:
        """Test that explicit dtype overrides precision context."""
        with precision_override(PrecisionStrategy.FULL_64):
            # Explicit dtype should override context
            result = load_array(npz_single_array["path"], dtype=torch.float16)
            assert result.dtype == torch.float16

    def test_load_multi_array_without_key_raises(self, npz_multi_array: dict) -> None:
        """Test that multi-array NPZ without key raises error through load_array."""
        with pytest.raises(ValueError, match="contains multiple arrays"):
            load_array(npz_multi_array["path"])

    def test_load_npz_preserves_data(self, npz_multi_array: dict) -> None:
        """Test that data is preserved correctly through NPZ loading."""
        features_tensor = load_array(npz_multi_array["path"], array_key="features")
        features_np = npz_multi_array["features"]

        # Convert to same dtype for comparison
        np.testing.assert_allclose(features_tensor.numpy(), features_np, rtol=1e-5, atol=1e-7)


class TestNpzIntegrationWithPrecision:
    """Integration tests for NPZ loading with precision system."""

    def test_mixed_precision_loading(self, npz_multi_array: dict) -> None:
        """Test loading different arrays with different precisions."""
        # Load features in float64
        with precision_override(PrecisionStrategy.FULL_64):
            features = load_array(npz_multi_array["path"], array_key="features")
            assert features.dtype == torch.float64

        # Load targets in float32
        with precision_override(PrecisionStrategy.FULL_32):
            targets = load_array(npz_multi_array["path"], array_key="targets")
            # Targets are int64, should be converted to float32
            assert targets.dtype == torch.float32

    def test_npz_with_int_arrays(self, npz_multi_array: dict) -> None:
        """Test NPZ loading with integer arrays respects precision conversion."""
        with precision_override(PrecisionStrategy.FULL_32):
            targets = load_array(npz_multi_array["path"], array_key="targets")
            # int64 array should be converted to float32 based on precision context
            assert targets.dtype == torch.float32

    def test_npz_array_key_forwarding(self, npz_multi_array: dict) -> None:
        """Test that array_key is properly forwarded through kwargs."""
        # This tests the **kwargs forwarding mechanism
        result = load_array(npz_multi_array["path"], dtype=torch.float64, array_key="latent")

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64
        assert result.shape == (10, 3)
