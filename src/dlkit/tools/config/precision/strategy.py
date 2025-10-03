"""Precision strategy enumeration with Lightning compatibility.

This module defines the core precision strategies available in DLKit,
with direct mapping to PyTorch Lightning precision modes.
"""

from __future__ import annotations

import torch
from enum import StrEnum
from typing import Literal


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
        """
        # Handle legacy integer values
        if precision == 64:
            precision = "64"
        elif precision == 32:
            precision = "32"
        elif precision == 16:
            precision = "16-mixed"

        lightning_to_strategy = {
            "64": cls.FULL_64,
            "64-true": cls.FULL_64,
            "32": cls.FULL_32,
            "32-true": cls.FULL_32,
            "16-mixed": cls.MIXED_16,
            "16": cls.TRUE_16,
            "16-true": cls.TRUE_16,
            "bf16-mixed": cls.MIXED_BF16,
            "bf16": cls.TRUE_BF16,
            "bf16-true": cls.TRUE_BF16,
        }

        if precision not in lightning_to_strategy:
            raise ValueError(
                f"Unsupported precision value: {precision}. "
                f"Supported values: {list(lightning_to_strategy.keys())}"
            )

        return lightning_to_strategy[precision]

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
