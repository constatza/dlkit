"""Centralized precision service following SRP and dependency inversion.

This module provides the core precision coordination service that manages
precision strategy resolution across all DLKit components.
"""

from __future__ import annotations

import torch
from typing import Any

from dlkit.tools.config.precision.strategy import PrecisionStrategy
from dlkit.interfaces.api.domain.precision import (
    PrecisionProvider,
    PrecisionContext,
    get_global_precision_context,
)


class PrecisionService:
    """Centralized precision coordination service following SRP.

    This service acts as the single source of truth for precision resolution
    throughout DLKit, coordinating between configuration, context overrides,
    and component-specific requirements.

    Responsibilities:
        - Resolve effective precision from multiple sources (config, context, defaults)
        - Provide consistent precision application across components
        - Convert between precision strategies and concrete types (torch.dtype, Lightning config)
        - Validate precision compatibility and constraints

    Design Principles:
        - Single Responsibility: Only handles precision coordination
        - Dependency Inversion: Depends on PrecisionProvider protocol, not concrete classes
        - Open/Closed: Extensible for new precision requirements without modification
    """

    def __init__(self, context: PrecisionContext | None = None) -> None:
        """Initialize precision service with optional context.

        Args:
            context: Precision context for override management.
                     If None, uses the global precision context.
        """
        self._context = context or get_global_precision_context()

    def resolve_precision(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> PrecisionStrategy:
        """Resolve effective precision from multiple sources.

        Resolution order (highest to lowest priority):
        1. Context override (thread-local API overrides)
        2. Provider precision (component-specific configuration)
        3. Explicit default parameter
        4. Global default (FULL_32)

        Args:
            provider: Optional precision provider (e.g., SessionSettings).
            default: Optional default precision if provider doesn't specify one.

        Returns:
            Resolved precision strategy to use.

        Examples:
            >>> service = PrecisionService()
            >>> # With context override
            >>> with precision_override(PrecisionStrategy.MIXED_16):
            ...     precision = service.resolve_precision()
            ...     assert precision == PrecisionStrategy.MIXED_16
        """
        # Priority 1: Context override (highest priority)
        context_override = self._context.get_override()
        if context_override is not None:
            return context_override

        # Priority 2: Provider precision
        if provider is not None:
            try:
                return provider.get_precision_strategy()
            except (AttributeError, NotImplementedError, RuntimeError):
                # Provider doesn't support precision or is broken, continue to next priority
                pass

        # Priority 3: Explicit default
        if default is not None:
            return default

        # Priority 4: Global default
        return PrecisionStrategy.get_default()

    def get_torch_dtype(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> torch.dtype:
        """Get torch.dtype for the resolved precision strategy.

        Args:
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            torch.dtype corresponding to the resolved precision.
        """
        precision = self.resolve_precision(provider, default)
        return precision.to_torch_dtype()

    def get_compute_dtype(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> torch.dtype:
        """Get computation dtype for the resolved precision strategy.

        For mixed precision, this returns float32 for computations.
        For true precision, this returns the native dtype.

        Args:
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            torch.dtype for computations and gradients.
        """
        precision = self.resolve_precision(provider, default)
        return precision.get_compute_dtype()

    def get_lightning_precision(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> str | int:
        """Get Lightning Trainer precision parameter for resolved strategy.

        Args:
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            Lightning-compatible precision value for Trainer configuration.
        """
        precision = self.resolve_precision(provider, default)
        return precision.to_lightning_precision()

    def is_mixed_precision(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> bool:
        """Check if resolved precision uses automatic mixed precision.

        Args:
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            True if the resolved precision uses autocast functionality.
        """
        precision = self.resolve_precision(provider, default)
        return precision.supports_autocast()

    def validate_precision_compatibility(
        self, precision: PrecisionStrategy, device_type: str = "cuda"
    ) -> bool:
        """Validate precision compatibility with device and PyTorch version.

        Args:
            precision: Precision strategy to validate.
            device_type: Device type ("cuda", "cpu", "mps", etc.).

        Returns:
            True if precision is compatible with the device and PyTorch version.

        Note:
            This is a placeholder for future validation logic.
            Currently returns True for all combinations.
        """
        # Future implementation could check:
        # - bfloat16 support on device
        # - PyTorch version compatibility
        # - Device-specific precision limitations
        return True

    def cast_tensor(
        self,
        tensor: torch.Tensor,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> torch.Tensor:
        """Cast tensor to the resolved precision dtype.

        Args:
            tensor: Input tensor to cast.
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            Tensor cast to the appropriate dtype.
        """
        target_dtype = self.get_torch_dtype(provider, default)
        return tensor.to(dtype=target_dtype)

    def apply_precision_to_model(
        self,
        model: torch.nn.Module,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> torch.nn.Module:
        """Apply precision strategy to model weights.

        Args:
            model: PyTorch model to apply precision to.
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            Model with weights cast to appropriate precision.
        """
        precision = self.resolve_precision(provider, default)
        target_dtype = precision.to_torch_dtype()
        model = model.to(dtype=target_dtype)

        # Sync precision metadata for DLKit models that track effective precision
        if hasattr(model, "_precision_dtype"):
            model._precision_dtype = target_dtype  # type: ignore[attr-defined]
        if hasattr(model, "_precision_applied"):
            model._precision_applied = True  # type: ignore[attr-defined]
        if hasattr(model, "_effective_precision_strategy"):
            model._effective_precision_strategy = precision  # type: ignore[attr-defined]

        return model

    def get_precision_info(
        self,
        provider: PrecisionProvider | None = None,
        default: PrecisionStrategy | None = None,
    ) -> dict[str, Any]:
        """Get comprehensive precision information for debugging/logging.

        Args:
            provider: Optional precision provider.
            default: Optional default precision strategy.

        Returns:
            Dictionary with precision details for debugging.
        """
        precision = self.resolve_precision(provider, default)
        return {
            "strategy": precision.name,
            "value": precision.value,
            "torch_dtype": str(precision.to_torch_dtype()),
            "compute_dtype": str(precision.get_compute_dtype()),
            "lightning_precision": precision.to_lightning_precision(),
            "supports_autocast": precision.supports_autocast(),
            "is_reduced_precision": precision.is_reduced_precision(),
            "memory_factor": precision.get_memory_factor(),
            "context_override": self._context.get_override(),
        }

    def infer_precision_from_model(self, model: torch.nn.Module) -> PrecisionStrategy | None:
        """Infer precision strategy from a model's parameter dtype.

        Inspects the model's parameters to determine the effective precision
        being used. This is useful for inferring precision from loaded checkpoints
        during inference.

        Args:
            model: PyTorch model to inspect.

        Returns:
            Inferred PrecisionStrategy based on model parameter dtype,
            or None if model has no parameters or dtype cannot be determined.

        Examples:
            >>> service = PrecisionService()
            >>> model = torch.nn.Linear(10, 5).to(torch.float32)
            >>> precision = service.infer_precision_from_model(model)
            >>> assert precision == PrecisionStrategy.FULL_32
        """
        try:
            # Get first parameter's dtype
            first_param = next(model.parameters(), None)
            if first_param is None:
                return None

            param_dtype = first_param.dtype

            # Map torch.dtype to PrecisionStrategy
            # For inference, we use "true" precision modes (not mixed)
            # since the model weights are already in the target dtype
            dtype_to_strategy = {
                torch.float64: PrecisionStrategy.FULL_64,
                torch.float32: PrecisionStrategy.FULL_32,
                torch.float16: PrecisionStrategy.TRUE_16,
                torch.bfloat16: PrecisionStrategy.TRUE_BF16,
            }

            return dtype_to_strategy.get(param_dtype)

        except (StopIteration, AttributeError):
            return None

    def __repr__(self) -> str:
        """String representation for debugging."""
        context_info = f"context_override={self._context.get_override()}"
        return f"PrecisionService({context_info})"


# Global precision service instance
_global_precision_service = PrecisionService()


def get_precision_service() -> PrecisionService:
    """Get the global precision service instance.

    Returns:
        Global PrecisionService instance for application-wide use.
    """
    return _global_precision_service
