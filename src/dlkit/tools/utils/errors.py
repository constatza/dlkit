"""Domain-specific errors for DLKit."""

from __future__ import annotations

from typing import Any


class DLKitError(Exception):
    """Base exception for all DLKit domain errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize DLKit error.

        Args:
            message: Error description
            context: Additional error context
        """
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
    """Model checkpoint loading error.

    Raised when model weights fail to load from checkpoint,
    such as state dict key mismatches or missing critical parameters.
    """


class PluginError(DLKitError):
    """Plugin configuration or execution error."""
