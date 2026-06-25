"""Segregated dependency protocols following Interface Segregation Principle.

This module splits the fat WorkflowDependencies interface into focused protocols
that clients can depend on only what they need, following ISP.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from dlkit.infrastructure.config import GeneralSettings

if TYPE_CHECKING:
    from loguru._logger import Logger
else:
    from loguru import logger as Logger


@runtime_checkable
class SettingsProvider(Protocol):
    """Protocol for components that need access to configuration settings."""

    @abstractmethod
    def get_settings(self) -> GeneralSettings:
        """Get the current configuration settings."""

    @abstractmethod
    def get_output_dir(self) -> Path | None:
        """Get output directory from settings."""

    @abstractmethod
    def get_data_dir(self) -> Path | None:
        """Get dataflow directory from settings."""


@runtime_checkable
class LoggingProvider(Protocol):
    """Protocol for components that need logging capabilities."""

    @abstractmethod
    def get_logger(self) -> Logger:
        """Get the configured logger instance."""


@runtime_checkable
class CheckpointProvider(Protocol):
    """Protocol for components that need checkpoint management."""

    @abstractmethod
    def get_checkpoint_path(self) -> Path | None:
        """Get checkpoint path with proper precedence."""

    @abstractmethod
    def with_checkpoint(self, checkpoint_path: Path) -> CheckpointProvider:
        """Create new provider with checkpoint path."""


@runtime_checkable
class ResourceProvider(Protocol):
    """Protocol for components that need temporary resources."""

    @abstractmethod
    def get_temp_dir(self) -> Path | None:
        """Get temporary directory path."""

    @abstractmethod
    def with_temp_dir(self, temp_dir: Path) -> ResourceProvider:
        """Create new provider with temp directory."""


@runtime_checkable
class MetadataProvider(Protocol):
    """Protocol for components that need metadata access."""

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Get workflow metadata."""

    @abstractmethod
    def get_metadata_value(self, key: str, default: Any = None) -> Any:
        """Get specific metadata value."""

    @abstractmethod
    def with_metadata(self, metadata: dict[str, Any]) -> MetadataProvider:
        """Create new provider with additional metadata."""


@runtime_checkable
class StrategyProvider(Protocol):
    """Protocol for components that need strategy information."""

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the current execution strategy name."""


@runtime_checkable
class OverrideProvider(Protocol):
    """Protocol for components that need runtime override tracking."""

    @abstractmethod
    def get_runtime_overrides(self) -> dict[str, Any]:
        """Get applied runtime overrides."""

    @abstractmethod
    def get_override_value(self, key: str, default: Any = None) -> Any:
        """Get specific runtime override value."""


# Composite protocols for common combinations
class ConfigurationContext(SettingsProvider, LoggingProvider, Protocol):
    """Protocol for components needing configuration and logging."""


class TrainingContext(
    SettingsProvider, LoggingProvider, CheckpointProvider, ResourceProvider, Protocol
):
    """Protocol for training-related components."""


class InferenceContext(SettingsProvider, LoggingProvider, CheckpointProvider, Protocol):
    """Protocol for inference-related components."""


class OptimizationContext(
    SettingsProvider, LoggingProvider, ResourceProvider, MetadataProvider, Protocol
):
    """Protocol for optimization-related components."""
