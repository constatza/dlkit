"""Tests for PrecisionStrategy enum."""

import pytest
import torch

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.tools.config.precision.strategy import _PRECISION_ALIAS_MAP


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
        with pytest.raises(ValueError, match="Invalid precision value"):
            PrecisionStrategy.from_lightning_precision("invalid")

        with pytest.raises(ValueError, match="Invalid Lightning precision integer"):
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


class TestPrecisionStringAliases:
    """Test suite for precision string alias support and validation.

    NOTE: These tests use ONLY semantic string aliases (no integers, no numeric strings).
    from_string() now rejects numeric strings like "64" - use "float64" or "double" instead.
    """

    def test_from_string_float64_aliases(self):
        """Test from_string with float64/double aliases."""
        assert PrecisionStrategy.from_string("double") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("float64") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("f64") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("fp64") == PrecisionStrategy.FULL_64

    def test_from_string_float32_aliases(self):
        """Test from_string with float32/single aliases."""
        assert PrecisionStrategy.from_string("single") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("float32") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("f32") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("fp32") == PrecisionStrategy.FULL_32

    def test_from_string_float16_aliases(self):
        """Test from_string with float16/half aliases."""
        assert PrecisionStrategy.from_string("half") == PrecisionStrategy.TRUE_16
        assert PrecisionStrategy.from_string("float16") == PrecisionStrategy.TRUE_16
        assert PrecisionStrategy.from_string("f16") == PrecisionStrategy.TRUE_16
        assert PrecisionStrategy.from_string("fp16") == PrecisionStrategy.TRUE_16

    def test_from_string_mixed16_aliases(self):
        """Test from_string with mixed16 aliases."""
        assert PrecisionStrategy.from_string("mixed16") == PrecisionStrategy.MIXED_16
        assert PrecisionStrategy.from_string("mixed_16") == PrecisionStrategy.MIXED_16

    def test_from_string_bfloat16_aliases(self):
        """Test from_string with bfloat16 aliases."""
        assert PrecisionStrategy.from_string("bfloat16") == PrecisionStrategy.TRUE_BF16

    def test_from_string_case_insensitive(self):
        """Test that from_string is case-insensitive."""
        assert PrecisionStrategy.from_string("DOUBLE") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("Double") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("FLOAT32") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("Single") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("HALF") == PrecisionStrategy.TRUE_16
        assert PrecisionStrategy.from_string("F64") == PrecisionStrategy.FULL_64

    def test_from_string_with_whitespace(self):
        """Test that from_string handles whitespace."""
        assert PrecisionStrategy.from_string(" double ") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_string("  single  ") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_string("  float64  ") == PrecisionStrategy.FULL_64

    def test_from_string_invalid_string(self):
        """Test that invalid strings raise helpful ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PrecisionStrategy.from_string("wtf")

        error_msg = str(exc_info.value)
        assert "Invalid precision value: 'wtf'" in error_msg
        assert "Supported aliases" in error_msg
        assert "Float64/Double:" in error_msg
        assert "Float32/Single:" in error_msg

    def test_from_string_rejects_numeric_strings(self):
        """Test that numeric strings are rejected."""
        invalid_numeric_strings = ["64", "32", "16"]

        for numeric_str in invalid_numeric_strings:
            with pytest.raises(ValueError, match="Invalid precision value"):
                PrecisionStrategy.from_string(numeric_str)

    def test_from_string_invalid_types(self):
        """Test that invalid precision values raise appropriate errors."""
        invalid_values = [
            "foo",
            "bar",
            "float128",
            "8",
            "mixed32",
            "true16",
            "fp8",
        ]

        for invalid_value in invalid_values:
            with pytest.raises(ValueError, match="Invalid precision value"):
                PrecisionStrategy.from_string(invalid_value)

    def test_from_lightning_precision_handles_integers(self):
        """Test that from_lightning_precision handles Lightning's integer format."""
        # from_lightning_precision should handle integers for backwards compat
        assert PrecisionStrategy.from_lightning_precision(64) == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_lightning_precision(32) == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_lightning_precision(16) == PrecisionStrategy.MIXED_16

    def test_from_lightning_precision_handles_numeric_strings(self):
        """Test that from_lightning_precision handles Lightning's numeric string format."""
        # from_lightning_precision should handle numeric strings like "64"
        assert PrecisionStrategy.from_lightning_precision("64") == PrecisionStrategy.FULL_64
        assert PrecisionStrategy.from_lightning_precision("32") == PrecisionStrategy.FULL_32
        assert PrecisionStrategy.from_lightning_precision("16-mixed") == PrecisionStrategy.MIXED_16

    def test_comprehensive_alias_coverage(self):
        """Test comprehensive coverage of all supported aliases.

        This test imports the authoritative alias map from the source module
        to ensure tests stay in sync with the actual implementation.
        """
        # Test all aliases from the authoritative mapping
        for alias, expected_strategy in _PRECISION_ALIAS_MAP.items():
            result = PrecisionStrategy.from_string(alias)
            assert result == expected_strategy, (
                f"Alias '{alias}' should map to {expected_strategy}, got {result}"
            )

    def test_alias_map_accessible(self):
        """Test that alias map is accessible via get_alias_map()."""
        alias_map = PrecisionStrategy.get_alias_map()

        # Verify it's the same as the imported map
        assert alias_map is _PRECISION_ALIAS_MAP

        # Verify some key aliases exist
        assert alias_map["double"] == PrecisionStrategy.FULL_64
        assert alias_map["single"] == PrecisionStrategy.FULL_32
        assert alias_map["half"] == PrecisionStrategy.TRUE_16
        assert alias_map["float64"] == PrecisionStrategy.FULL_64

        # Verify numeric strings and integers are NOT in the map
        assert "64" not in alias_map
        assert "32" not in alias_map
        assert 64 not in alias_map
        assert 32 not in alias_map
