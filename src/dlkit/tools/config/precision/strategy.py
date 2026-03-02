"""Precision strategy enumeration with Lightning compatibility.

This module defines the core precision strategies available in DLKit,
with direct mapping to PyTorch Lightning precision modes.
"""

from __future__ import annotations

import torch
from enum import StrEnum
from typing import Literal, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Final


class PrecisionStrategy(StrEnum):
    """Precision strategies with direct Lightning Trainer compatibility.

    This enum provides precision modes that map directly to PyTorch Lightning's
    precision parameter while maintaining clear semantic meaning for DLKit users.

    The enum inherits from str to maintain JSON/TOML serialization compatibility
    and provides seamless integration with Pydantic settings.

    Precision Modes:
        FULL_64: Double precision (64-bit) for maximum numerical accuracy
        FULL_32: Full precision (32-bit) - default for safety and compatibility
        MIXED_16: Mixed precision (16-bit/32-bit) for memory efficiency
        TRUE_16: True half precision (16-bit) for maximum memory savings
        MIXED_BF16: Mixed bfloat16 precision for improved gradient stability
        TRUE_BF16: True bfloat16 precision for memory efficiency with stability
    """

    # 64-bit precision for maximum accuracy
    FULL_64 = "64"

    # 32-bit precision (default for safety)
    FULL_32 = "32"

    # 16-bit mixed precision for memory efficiency
    MIXED_16 = "16-mixed"

    # 16-bit true precision for maximum memory savings
    TRUE_16 = "16"

    # bfloat16 mixed precision for improved stability
    MIXED_BF16 = "bf16-mixed"

    # bfloat16 true precision for memory efficiency
    TRUE_BF16 = "bf16"

    def to_lightning_precision(self) -> str | int:
        """Convert to PyTorch Lightning Trainer precision parameter.

        Returns:
            Lightning-compatible precision value for Trainer configuration.

        Examples:
            >>> PrecisionStrategy.FULL_32.to_lightning_precision()
            32
            >>> PrecisionStrategy.MIXED_16.to_lightning_precision()
            "16-mixed"
        """
        lightning_map = {
            PrecisionStrategy.FULL_64: 64,
            PrecisionStrategy.FULL_32: 32,
            PrecisionStrategy.MIXED_16: "16-mixed",
            PrecisionStrategy.TRUE_16: "16-true",
            PrecisionStrategy.MIXED_BF16: "bf16-mixed",
            PrecisionStrategy.TRUE_BF16: "bf16-true",
        }
        return lightning_map[self]

    def to_torch_dtype(self) -> torch.dtype:
        """Convert to corresponding torch.dtype for model weights and tensors.

        Returns:
            Primary torch.dtype for this precision strategy.
            Note: Mixed precision strategies return the lower precision dtype
            as the primary type, with automatic upcasting handled by Lightning.

        Examples:
            >>> PrecisionStrategy.FULL_32.to_torch_dtype()
            torch.float32
            >>> PrecisionStrategy.MIXED_16.to_torch_dtype()
            torch.float16
        """
        dtype_map = {
            PrecisionStrategy.FULL_64: torch.float64,
            PrecisionStrategy.FULL_32: torch.float32,
            PrecisionStrategy.MIXED_16: torch.float16,
            PrecisionStrategy.TRUE_16: torch.float16,
            PrecisionStrategy.MIXED_BF16: torch.bfloat16,
            PrecisionStrategy.TRUE_BF16: torch.bfloat16,
        }
        return dtype_map[self]

    def get_compute_dtype(self) -> torch.dtype:
        """Get the dtype used for computation and gradients.

        For mixed precision strategies, this returns float32 as computations
        are performed in higher precision. For true precision strategies,
        this returns the native dtype.

        Returns:
            torch.dtype used for actual computations and gradient calculations.
        """
        if self in (PrecisionStrategy.MIXED_16, PrecisionStrategy.MIXED_BF16):
            return torch.float32
        return self.to_torch_dtype()

    def supports_autocast(self) -> bool:
        """Check if this precision strategy uses automatic mixed precision.

        Returns:
            True if this strategy uses PyTorch's autocast functionality.
        """
        return self in (PrecisionStrategy.MIXED_16, PrecisionStrategy.MIXED_BF16)

    def is_reduced_precision(self) -> bool:
        """Check if this strategy uses reduced precision (< 32-bit).

        Returns:
            True if this strategy uses less than 32-bit precision.
        """
        return self != PrecisionStrategy.FULL_64 and self != PrecisionStrategy.FULL_32

    def get_memory_factor(self) -> float:
        """Estimate memory usage factor relative to FULL_32.

        Returns:
            Approximate memory usage multiplier compared to 32-bit precision.
            Values < 1.0 indicate memory savings, > 1.0 indicate increased usage.
        """
        memory_factors = {
            PrecisionStrategy.FULL_64: 2.0,  # Double memory usage
            PrecisionStrategy.FULL_32: 1.0,  # Baseline
            PrecisionStrategy.MIXED_16: 0.7,  # ~30% savings (mixed usage)
            PrecisionStrategy.TRUE_16: 0.5,  # ~50% savings
            PrecisionStrategy.MIXED_BF16: 0.7,  # ~30% savings (mixed usage)
            PrecisionStrategy.TRUE_BF16: 0.5,  # ~50% savings
        }
        return memory_factors[self]

    @classmethod
    def get_alias_map(cls) -> Mapping[str, "PrecisionStrategy"]:
        """Get the comprehensive alias mapping for precision values.

        This method returns the authoritative mapping of all supported precision
        formats to their corresponding PrecisionStrategy enum values. This mapping
        is used by from_string() for validation and can be imported by tests.

        Returns:
            Immutable mapping of string aliases to PrecisionStrategy values.
        """
        return _PRECISION_ALIAS_MAP

    @classmethod
    def from_string(cls, value: str) -> PrecisionStrategy:
        """Create PrecisionStrategy from string with support for common aliases.

        This method accepts various precision format strings and normalizes them
        to the appropriate PrecisionStrategy. Only semantic string aliases are supported
        (no integers, no numeric strings like "64").

        Supported formats:
            - Float64/Double: "double", "float64", "f64", "fp64"
            - Float32/Single: "single", "float32", "f32", "fp32"
            - Float16/Half: "half", "float16", "f16", "fp16"
            - Mixed16: "mixed16", "mixed_16"
            - BFloat16: "bfloat16", "bf16"
            - BFloat16 Mixed: "bfloat16_mixed", "bf16_mixed"

        Args:
            value: Precision string to parse (case-insensitive).

        Returns:
            Corresponding PrecisionStrategy enum value.

        Raises:
            ValueError: If precision value is not recognized. Error message includes
                       all supported formats.

        Examples:
            >>> PrecisionStrategy.from_string("double")
            PrecisionStrategy.FULL_64
            >>> PrecisionStrategy.from_string("float32")
            PrecisionStrategy.FULL_32
            >>> PrecisionStrategy.from_string("half")
            PrecisionStrategy.TRUE_16
            >>> PrecisionStrategy.from_string("mixed16")
            PrecisionStrategy.MIXED_16
            >>> PrecisionStrategy.from_string("wtf")  # doctest: +SKIP
            ValueError: Invalid precision value: 'wtf'...
        """
        # Normalize string to lowercase for case-insensitive matching
        value_lower = value.lower().strip()

        if value_lower in _PRECISION_ALIAS_MAP:
            return _PRECISION_ALIAS_MAP[value_lower]

        # Value not recognized - provide helpful error message
        raise ValueError(
            f"Invalid precision value: '{value}'. "
            f"Supported aliases (case-insensitive):\n"
            f"  - Float64/Double: double, float64, f64, fp64\n"
            f"  - Float32/Single: single, float32, f32, fp32\n"
            f"  - Float16/Half: half, float16, f16, fp16\n"
            f"  - Mixed16: mixed16, mixed_16\n"
            f"  - BFloat16: bfloat16, bf16\n"
            f"  - BFloat16 Mixed: bfloat16_mixed, bf16_mixed"
        )

    @classmethod
    def from_lightning_precision(
        cls,
        precision: str
        | int
        | Literal[
            "64",
            "64-true",
            "32",
            "32-true",
            "16-mixed",
            "16",
            "16-true",
            "bf16-mixed",
            "bf16",
            "bf16-true",
        ],
    ) -> PrecisionStrategy:
        """Create PrecisionStrategy from Lightning Trainer precision value.

        This method handles Lightning's numeric integer format for backwards
        compatibility (64, 32, 16) and converts them to semantic strings.

        Args:
            precision: Lightning Trainer precision parameter value.

        Returns:
            Corresponding PrecisionStrategy enum value.

        Raises:
            ValueError: If precision value is not supported.

        Examples:
            >>> PrecisionStrategy.from_lightning_precision("16-mixed")
            PrecisionStrategy.MIXED_16
            >>> PrecisionStrategy.from_lightning_precision(32)
            PrecisionStrategy.FULL_32

        Note:
            This method is maintained for backwards compatibility with Lightning.
            New code should use from_string() which supports semantic aliases.
        """
        # Handle Legacy integer values from Lightning
        if isinstance(precision, int):
            if precision == 64:
                return cls.FULL_64
            elif precision == 32:
                return cls.FULL_32
            elif precision == 16:
                return cls.MIXED_16  # Default to mixed for integer 16
            else:
                raise ValueError(
                    f"Invalid Lightning precision integer: {precision}. Supported: 64, 32, 16"
                )

        # Handle Lightning string formats (including numeric strings)
        value_str = str(precision).lower().strip()

        # Map Lightning numeric strings to semantic aliases
        lightning_numeric_map = {
            "64": "float64",
            "64-true": "float64",
            "32": "float32",
            "32-true": "float32",
            "16": "half",
            "16-true": "half",
            "16-mixed": "mixed16",
            "bf16": "bfloat16",
            "bf16-true": "bfloat16",
            "bf16-mixed": "bfloat16_mixed",
        }

        if value_str in lightning_numeric_map:
            return cls.from_string(lightning_numeric_map[value_str])

        # Try direct lookup for semantic strings
        return cls.from_string(value_str)

    @classmethod
    def get_default(cls) -> PrecisionStrategy:
        """Get the default precision strategy.

        Returns:
            FULL_32 as the default for safety and compatibility.
        """
        return cls.FULL_32

    def __str__(self) -> str:
        """String representation using the enum value."""
        return self.value

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return f"PrecisionStrategy.{self.name}('{self.value}')"


# Authoritative precision alias mapping - single source of truth
# This mapping is used by from_string() and can be imported by tests
# ONLY semantic string aliases - no integers, no numeric strings
_PRECISION_ALIAS_MAP: "Final[Mapping[str, PrecisionStrategy]]" = {
    # Float64/Double precision aliases
    "double": PrecisionStrategy.FULL_64,
    "float64": PrecisionStrategy.FULL_64,
    "f64": PrecisionStrategy.FULL_64,
    "fp64": PrecisionStrategy.FULL_64,
    # Float32/Single precision aliases
    "single": PrecisionStrategy.FULL_32,
    "float32": PrecisionStrategy.FULL_32,
    "f32": PrecisionStrategy.FULL_32,
    "fp32": PrecisionStrategy.FULL_32,
    # Float16/Half precision aliases
    "half": PrecisionStrategy.TRUE_16,
    "float16": PrecisionStrategy.TRUE_16,
    "f16": PrecisionStrategy.TRUE_16,
    "fp16": PrecisionStrategy.TRUE_16,
    # Mixed16 aliases
    "mixed16": PrecisionStrategy.MIXED_16,
    "mixed_16": PrecisionStrategy.MIXED_16,
    # BFloat16 true precision aliases
    "bfloat16": PrecisionStrategy.TRUE_BF16,
    "bf16": PrecisionStrategy.TRUE_BF16,
    # BFloat16 mixed precision aliases
    "bfloat16_mixed": PrecisionStrategy.MIXED_BF16,
    "bf16_mixed": PrecisionStrategy.MIXED_BF16,
}
