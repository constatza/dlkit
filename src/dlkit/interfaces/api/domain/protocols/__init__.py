"""Domain protocols for clean architecture interfaces."""

from abc import abstractmethod

# Define minimal protocols to avoid circular imports
from typing import Protocol

from .dependency_protocols import (
    CheckpointProvider,
    ConfigurationContext,
    InferenceContext,
    LoggingProvider,
    MetadataProvider,
    OptimizationContext,
    OverrideProvider,
    ResourceProvider,
    SettingsProvider,
    StrategyProvider,
    TrainingContext,
)


class ExecutionStrategy(Protocol):
    """Protocol for execution strategies."""

    @abstractmethod
    def execute(self, context):
        """Execute the strategy with given context."""
        ...

    @abstractmethod
    def validate_config(self, settings):
        """Validate configuration for this strategy."""
        ...


class StrategyFactory(Protocol):
    """Protocol for strategy factory implementations."""

    @abstractmethod
    def create_strategy(self, mode: str):
        """Create execution strategy for given mode."""
        ...

    @abstractmethod
    def get_supported_modes(self):
        """Get list of supported execution modes."""
        ...


class WorkflowOperation(Protocol):
    """Protocol for atomic workflow operations."""

    @abstractmethod
    def execute(self, context):
        """Execute the operation with given context."""
        ...


__all__ = [
    # Core protocols
    "ExecutionStrategy",
    "StrategyFactory",
    "WorkflowOperation",
    # Dependency protocols
    "SettingsProvider",
    "LoggingProvider",
    "CheckpointProvider",
    "ResourceProvider",
    "MetadataProvider",
    "StrategyProvider",
    "OverrideProvider",
    "ConfigurationContext",
    "TrainingContext",
    "InferenceContext",
    "OptimizationContext",
]
