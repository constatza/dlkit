"""Tests for PrecisionStrategy enum."""

import pytest
import torch

from dlkit.tools.config.precision import PrecisionStrategy


class TestPrecisionStrategy:
    """Test suite for PrecisionStrategy enum."""

    def test_default_strategy(self):
        """Test that default strategy is FULL_32."""
        default = PrecisionStrategy.get_default()
        assert default == PrecisionStrategy.FULL_32

    def test_lightning_precision_conversion(self):
        """Test conversion to Lightning precision parameters."""
        assert PrecisionStrategy.FULL_64.to_lightning_precision() == 64
        assert PrecisionStrategy.FULL_32.to_lightning_precision() == 32
        assert PrecisionStrategy.MIXED_16.to_lightning_precision() == "16-mixed"
        assert PrecisionStrategy.TRUE_16.to_lightning_precision() == "16-true"
        assert PrecisionStrategy.MIXED_BF16.to_lightning_precision() == "bf16-mixed"
        assert PrecisionStrategy.TRUE_BF16.to_lightning_precision() == "bf16-true"

    def test_torch_dtype_conversion(self):
        """Test conversion to torch.dtype."""
        assert PrecisionStrategy.FULL_64.to_torch_dtype() == torch.float64
        assert PrecisionStrategy.FULL_32.to_torch_dtype() == torch.float32
        assert PrecisionStrategy.MIXED_16.to_torch_dtype() == torch.float16
        assert PrecisionStrategy.TRUE_16.to_torch_dtype() == torch.float16
        assert PrecisionStrategy.MIXED_BF16.to_torch_dtype() == torch.bfloat16
        assert PrecisionStrategy.TRUE_BF16.to_torch_dtype() == torch.bfloat16

    def test_compute_dtype(self):
        """Test compute dtype resolution for mixed vs true precision."""
        # Mixed precision uses float32 for computation
        assert PrecisionStrategy.MIXED_16.get_compute_dtype() == torch.float32
        assert PrecisionStrategy.MIXED_BF16.get_compute_dtype() == torch.float32

        # True precision uses native dtype
        assert PrecisionStrategy.FULL_64.get_compute_dtype() == torch.float64
        assert PrecisionStrategy.FULL_32.get_compute_dtype() == torch.float32
        assert PrecisionStrategy.TRUE_16.get_compute_dtype() == torch.float16
        assert PrecisionStrategy.TRUE_BF16.get_compute_dtype() == torch.bfloat16

    def test_autocast_support(self):
        """Test autocast support detection."""
        # Mixed precision supports autocast
        assert PrecisionStrategy.MIXED_16.supports_autocast()
        assert PrecisionStrategy.MIXED_BF16.supports_autocast()

        # True precision does not use autocast
        assert not PrecisionStrategy.FULL_64.supports_autocast()
        assert not PrecisionStrategy.FULL_32.supports_autocast()
        assert not PrecisionStrategy.TRUE_16.supports_autocast()
        assert not PrecisionStrategy.TRUE_BF16.supports_autocast()

    def test_reduced_precision_detection(self):
        """Test reduced precision detection."""
        # Full precision strategies
        assert not PrecisionStrategy.FULL_64.is_reduced_precision()
        assert not PrecisionStrategy.FULL_32.is_reduced_precision()

        # Reduced precision strategies
        assert PrecisionStrategy.MIXED_16.is_reduced_precision()
        assert PrecisionStrategy.TRUE_16.is_reduced_precision()
        assert PrecisionStrategy.MIXED_BF16.is_reduced_precision()
        assert PrecisionStrategy.TRUE_BF16.is_reduced_precision()

    def test_memory_factors(self):
        """Test memory usage factor estimates."""
        assert PrecisionStrategy.FULL_64.get_memory_factor() == 2.0
        assert PrecisionStrategy.FULL_32.get_memory_factor() == 1.0
        assert PrecisionStrategy.MIXED_16.get_memory_factor() == 0.7
        assert PrecisionStrategy.TRUE_16.get_memory_factor() == 0.5
        assert PrecisionStrategy.MIXED_BF16.get_memory_factor() == 0.7
        assert PrecisionStrategy.TRUE_BF16.get_memory_factor() == 0.5

    def test_from_lightning_precision(self):
        """Test creation from Lightning precision values."""
        # String values
        assert PrecisionStrategy.from_lightning_precision("64") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_lightning_precision("32") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_lightning_precision("16-mixed") == PrecisionStrategy.MIXED_16
        assert PrecisionStrategy.from_lightning_precision("16") == PrecisionStrategy.TRUE_16
        assert (
            PrecisionStrategy.from_lightning_precision("bf16-mixed") == PrecisionStrategy.MIXED_BF16
        )
        assert PrecisionStrategy.from_lightning_precision("bf16") == PrecisionStrategy.TRUE_BF16

        # Legacy integer values
        assert PrecisionStrategy.from_lightning_precision(64) == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_lightning_precision(32) == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_lightning_precision(16) == PrecisionStrategy.MIXED_16
        assert PrecisionStrategy.from_lightning_precision("16-true") == PrecisionStrategy.TRUE_16
        assert PrecisionStrategy.from_lightning_precision("64-true") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_lightning_precision("32-true") == PrecisionStrategy.FULL_32
        assert (
            PrecisionStrategy.from_lightning_precision("bf16-true") == PrecisionStrategy.TRUE_BF16
        )

    def test_from_lightning_precision_invalid(self):
        """Test invalid Lightning precision values raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported precision value"):
            PrecisionStrategy.from_lightning_precision("invalid")

        with pytest.raises(ValueError, match="Unsupported precision value"):
            PrecisionStrategy.from_lightning_precision(42)

    def test_string_representation(self):
        """Test string representations."""
        assert str(PrecisionStrategy.FULL_32) == "32"
        assert repr(PrecisionStrategy.MIXED_16) == "PrecisionStrategy.MIXED_16('16-mixed')"

    def test_enum_value_consistency(self):
        """Test that enum values match Lightning precision strings."""
        for strategy in PrecisionStrategy:
            assert strategy.value == str(strategy)

        assert PrecisionStrategy.FULL_64.to_lightning_precision() == 64
        assert PrecisionStrategy.FULL_32.to_lightning_precision() == 32
        assert PrecisionStrategy.TRUE_16.to_lightning_precision() == "16-true"
        assert PrecisionStrategy.TRUE_BF16.to_lightning_precision() == "bf16-true"
