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
