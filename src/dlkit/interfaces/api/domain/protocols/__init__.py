"""Domain protocols for clean architecture interfaces."""

from abc import abstractmethod

# Define minimal protocols to avoid circular imports
from typing import Protocol, runtime_checkable

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
from .workflow_executor import IWorkflowExecutor


@runtime_checkable
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


@runtime_checkable
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


@runtime_checkable
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
    "IWorkflowExecutor",
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
