"""Shared error hierarchy for DLKit."""

from __future__ import annotations

from typing import Any


class DLKitError(Exception):
    """Base exception for all DLKit errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    @property
    def correlation_id(self) -> str | None:
        """Get correlation ID for error tracking."""
        return self.context.get("correlation_id")


class ConfigurationError(DLKitError):
    """Configuration validation or loading error."""


class WorkflowError(DLKitError):
    """Workflow execution error."""


class StrategyError(DLKitError):
    """Strategy selection or execution error."""


class ModelStateError(DLKitError):
    """Model state construction or management error."""


class ModelLoadingError(DLKitError):
    """Model checkpoint loading error."""


class ForwardContractError(DLKitError):
    """Raised when a model's declared entry-name contract cannot be
    reconciled with its forward() signature, or when caller-supplied kwargs
    at inference time don't match the checkpoint's persisted forward-arg
    contract."""


class PluginError(DLKitError):
    """Plugin configuration or execution error."""


class OptimizerPolicyError(DLKitError):
    """Raised when an optimization program cannot be built or executed."""


class ParameterPartitionError(DLKitError):
    """Raised when parameter partitioning produces overlapping or invalid groups."""


class StageTransitionError(DLKitError):
    """Raised when a stage transition cannot be completed."""


class UnsupportedOptimizerPolicyError(DLKitError):
    """Raised when an optimization program configuration is not supported."""


class PlaceholderNotResolvedError(ValueError):
    """Raised when a placeholder entry is used without value injection."""

    def __init__(self, entry_name: str) -> None:
        """Initialize with entry name.

        Args:
            entry_name: Name of the unresolved placeholder entry
        """
        super().__init__(
            f"Entry '{entry_name}' is a placeholder without path or value. "
            f"Either specify 'path' in config or inject 'value' programmatically."
        )


class ConfigValidationError(ValueError):
    """Raised when configuration is incomplete, invalid, or fails validation."""

    def __init__(
        self,
        message: str,
        *,
        model_class: str = "",
        section_data: dict[str, str] | None = None,
    ) -> None:
        """Initialize with a message and optional validation context.

        Args:
            message: Human-readable description of the validation failure.
            model_class: Name of the Pydantic model that failed validation, if any.
            section_data: Raw section data that failed validation, if any.
        """
        super().__init__(message)
        self.model_class = model_class
        self.section_data = section_data or {}


class BatchComplianceError(ValueError):
    """Raised when dataset sources violate batch-shape invariants."""
