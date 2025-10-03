"""Application layer for optimization services.

This package contains application services that orchestrate optimization workflows
by coordinating between domain models, repositories, and infrastructure adapters.

Each service follows the Single Responsibility Principle:
- StudyManager: Study lifecycle management
- TrialExecutor: Individual trial execution
- OptimizationOrchestrator: Complete optimization workflow coordination

The services depend on abstractions (domain protocols) rather than concrete
implementations, following the Dependency Inversion Principle.
"""

from .services import (
    StudyManager,
    TrialExecutor,
    OptimizationOrchestrator,
)

__all__ = [
    "StudyManager",
    "TrialExecutor",
    "OptimizationOrchestrator",
]
