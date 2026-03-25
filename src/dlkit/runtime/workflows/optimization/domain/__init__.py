"""Domain layer for optimization following Domain-Driven Design principles.

This package contains pure business logic and domain models without any
infrastructure dependencies. It represents the core optimization concepts:

- Study: An optimization session (aggregate root)
- Trial: Individual hyperparameter attempts within a Study
- OptimizationResult: Complete optimization outcome

The domain layer follows SOLID principles:
- Single Responsibility: Each model has one clear purpose
- Open/Closed: New optimization strategies can be added via protocols
- Liskov Substitution: All implementations can be substituted
- Interface Segregation: Focused, cohesive interfaces
- Dependency Inversion: Depends on abstractions, not concretions
"""

from .models import (
    HyperParameter,
    ITrialObjective,
    OptimizationDirection,
    OptimizationResult,
    Study,
    Trial,
    TrialFailedException,
    TrialPrunedException,
    TrialState,
)
from .protocols import (
    IConfigurationPersistence,
    IExperimentTracker,
    IHyperparameterSampler,
    IObjectiveFunction,
    IPruningStrategy,
    IStudyRepository,
    IStudyRunContext,
    ITrialExecutor,
    ITrialRunContext,
)

__all__ = [
    # Domain Models
    "Study",
    "Trial",
    "OptimizationResult",
    "OptimizationDirection",
    "TrialState",
    "HyperParameter",
    "ITrialObjective",
    "TrialPrunedException",
    "TrialFailedException",
    # Domain Protocols
    "IStudyRepository",
    "IHyperparameterSampler",
    "IPruningStrategy",
    "IExperimentTracker",
    "IStudyRunContext",
    "ITrialRunContext",
    "IConfigurationPersistence",
    "ITrialExecutor",
    "IObjectiveFunction",
]
