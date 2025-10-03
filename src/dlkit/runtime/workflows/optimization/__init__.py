"""Clean optimization architecture following Domain-Driven Design.

This package provides a complete optimization solution that properly models
the Optuna Study → Trial hierarchy and integrates cleanly with MLflow tracking.

Architecture Layers:
1. Domain: Pure business logic and models (Study, Trial, OptimizationResult)
2. Application: Services that orchestrate workflows (OptimizationOrchestrator)
3. Infrastructure: Adapters for external services (Optuna, MLflow)

Key Features:
- Proper SOLID principles compliance
- Clean separation of concerns
- Dependency inversion with protocol-based design
- Proper nested MLflow run structure
- Testable, composable services

Usage:
    from dlkit.runtime.workflows.optimization import create_optimization_orchestrator

    orchestrator = create_optimization_orchestrator(settings)
    result = orchestrator.execute_optimization(
        study_name="my_study",
        base_settings=settings,
        n_trials=100,
        direction=OptimizationDirection.MINIMIZE
    )
"""

# Domain exports
from .domain import (
    Study,
    Trial,
    OptimizationResult,
    OptimizationDirection,
    TrialState,
    HyperParameter,
    IStudyRepository,
    IExperimentTracker,
    IConfigurationPersistence,
    ITrialExecutor,
    TrialPrunedException,
    TrialFailedException,
)

# Application services exports
from .application import (
    StudyManager,
    TrialExecutor,
    OptimizationOrchestrator,
)

# Infrastructure exports
from .infrastructure import (
    OptunaStudyRepository,
    InMemoryStudyRepository,
    MLflowTrackingAdapter,
    NullTrackingAdapter,
    TOMLConfigurationPersister,
    JSONConfigurationPersister,
    NullConfigurationPersister,
)

# Factory and Strategy exports
from .factory import (
    OptimizationServiceFactory,
)
from .strategy import OptimizationStrategy

__all__ = [
    # Domain Models
    "Study",
    "Trial",
    "OptimizationResult",
    "OptimizationDirection",
    "TrialState",
    "HyperParameter",
    # Domain Protocols
    "IStudyRepository",
    "IExperimentTracker",
    "IConfigurationPersistence",
    "ITrialExecutor",
    "TrialPrunedException",
    "TrialFailedException",
    # Application Services
    "StudyManager",
    "TrialExecutor",
    "OptimizationOrchestrator",
    # Infrastructure
    "OptunaStudyRepository",
    "InMemoryStudyRepository",
    "MLflowTrackingAdapter",
    "NullTrackingAdapter",
    "TOMLConfigurationPersister",
    "JSONConfigurationPersister",
    "NullConfigurationPersister",
    # Factory and Strategy
    "OptimizationServiceFactory",
    "OptimizationStrategy",
]
