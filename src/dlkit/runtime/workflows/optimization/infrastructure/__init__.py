"""Infrastructure layer for optimization.

This package contains adapters that implement domain protocols using
external libraries and services. It follows the Dependency Inversion
Principle by depending on domain abstractions rather than concrete implementations.

Infrastructure Components:
- Repositories: Study persistence using Optuna
- Tracking: Experiment tracking using MLflow
- Persistence: Configuration file persistence
- Adapters: External service integrations
"""

from .repositories import (
    OptunaStudyRepository,
    InMemoryStudyRepository,
)

from .tracking import (
    MLflowTrackingAdapter,
    NullTrackingAdapter,
)

from .persistence import (
    TOMLConfigurationPersister,
    JSONConfigurationPersister,
    NullConfigurationPersister,
    FileSystemConfigurationPersister,
)

__all__ = [
    # Repositories
    "OptunaStudyRepository",
    "InMemoryStudyRepository",
    # Tracking
    "MLflowTrackingAdapter",
    "NullTrackingAdapter",
    # Persistence
    "TOMLConfigurationPersister",
    "JSONConfigurationPersister",
    "NullConfigurationPersister",
    "FileSystemConfigurationPersister",
]
