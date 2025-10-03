# Clean Optimization Architecture

This package provides a complete refactoring of the DLKit optimization system, fixing critical SOLID principle violations and implementing proper Domain-Driven Design.

## Problems with Previous Architecture

The previous optimization implementation had severe architectural issues:

1. **SOLID Violations**:
   - `OptunaOptimization`: 293-line God Object mixing optimization, MLflow, I/O, and configuration
   - Mixed responsibilities preventing proper composition
   - Direct dependencies on concrete implementations (Optuna, MLflow)
   - Fat interfaces with unclear responsibilities

2. **"Random Classes" Problem**:
   - 4 different overlapping optimization implementations
   - `OptunaOptimization`, `OptimizationDecorator`, `OptimizationAdapter`, `OptunaOptimizer`
   - Confusing routing logic in factory

3. **Missing Optuna Hierarchy**:
   - Tests expected "nested runs" but architecture didn't model Study → Trial hierarchy
   - MLflow runs didn't properly reflect Optuna's Study/Trial relationship
   - No clear separation between study-level and trial-level concerns

## New Clean Architecture

### Domain Layer (`domain/`)

Pure business logic with no infrastructure dependencies:

```python
# Core domain models
Study          # Optimization session (aggregate root)
Trial          # Individual hyperparameter attempt
OptimizationResult  # Complete optimization outcome

# Domain protocols (following DIP)
IStudyRepository          # Study persistence abstraction
IExperimentTracker       # Experiment tracking abstraction
IConfigurationPersistence # Configuration storage abstraction
```

### Application Layer (`application/`)

Services that orchestrate workflows:

```python
StudyManager              # Study lifecycle management
TrialExecutor            # Individual trial execution
OptimizationOrchestrator # Complete workflow coordination
```

### Infrastructure Layer (`infrastructure/`)

Adapters for external services:

```python
OptunaStudyRepository     # Optuna-based study storage
MLflowTrackingAdapter    # MLflow experiment tracking
TOMLConfigurationPersister # TOML config persistence
```

## Key Architectural Improvements

### 1. Proper Optuna Study → Trial Hierarchy

The new architecture correctly models Optuna's conceptual hierarchy:

```python
Study (contains multiple) → Trial → Best Retrain
```

This creates proper nested MLflow run structure:
- **Parent Run**: Optimization study metadata
- **Child Runs**: Individual trial executions
- **Final Child Run**: Best parameter retraining

### 2. SOLID Principles Compliance

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: New strategies added via protocols, no modification
- **Liskov Substitution**: All implementations are interchangeable
- **Interface Segregation**: Focused, cohesive interfaces
- **Dependency Inversion**: Depends on abstractions, not concretions

### 3. Clean Dependency Injection

```python
factory = OptimizationServiceFactory(
    study_repository=OptunaStudyRepository(),
    experiment_tracker=MLflowTrackingAdapter(),
    config_persister=TOMLConfigurationPersister()
)

orchestrator = factory.create_optimization_orchestrator(settings)
```

### 4. Proper Separation of Concerns

Each component focuses on a single concern:
- **Domain models**: Pure business logic
- **Application services**: Workflow orchestration
- **Infrastructure adapters**: External service integration

## Usage

### Basic Usage

```python
from dlkit.runtime.workflows.optimization import create_optimization_orchestrator

orchestrator = create_optimization_orchestrator(settings)
result = orchestrator.execute_optimization(
    study_name="hyperparameter_search",
    base_settings=settings,
    n_trials=100,
    direction=OptimizationDirection.MINIMIZE
)
```

### Advanced Usage with Dependency Injection

```python
from dlkit.runtime.workflows.optimization import (
    OptimizationServiceFactory,
    MLflowTrackingAdapter,
    TOMLConfigurationPersister
)

factory = OptimizationServiceFactory(
    experiment_tracker=MLflowTrackingAdapter(),
    config_persister=TOMLConfigurationPersister()
)

orchestrator = factory.create_optimization_orchestrator(settings)
result = orchestrator.execute_optimization(...)
```

## Migration Guide

### From Old Architecture

Replace:
```python
# OLD: God Object with SOLID violations
from dlkit.runtime.workflows.strategies.optimization import OptunaOptimization
strategy = OptunaOptimization()
result = strategy.run(settings)
```

With:
```python
# NEW: Clean architecture with proper separation
from dlkit.runtime.workflows.optimization import create_optimization_orchestrator
orchestrator = create_optimization_orchestrator(settings)
result = orchestrator.execute_optimization(
    study_name="my_study",
    base_settings=settings,
    n_trials=100,
    direction=OptimizationDirection.MINIMIZE
)
```

### Factory Integration

The existing factory now routes to the clean architecture:

```python
# This automatically uses CleanOptimizationStrategy
factory = ExecutionStrategyFactory()
strategy = factory.create_optimization_strategy(settings)
result = strategy.execute_optimization(settings)
```

## Testing

The new architecture is highly testable with dependency injection:

```python
from dlkit.runtime.workflows.optimization.infrastructure import InMemoryStudyRepository
from dlkit.runtime.workflows.optimization.application import StudyManager

# Test with in-memory repository
repository = InMemoryStudyRepository()
manager = StudyManager(repository)
study = manager.create_study("test_study", OptimizationDirection.MINIMIZE, 10)
```

## Benefits

1. ✅ **SOLID Compliance**: Each class has single responsibility
2. ✅ **Proper Hierarchy**: Study → Trial → Best Retrain with nested MLflow runs
3. ✅ **Testability**: Easy to test with dependency injection
4. ✅ **Maintainability**: Clear separation of concerns
5. ✅ **Extensibility**: Easy to add new optimization strategies
6. ✅ **No More "Random Classes"**: Clear, purposeful architecture

The new architecture eliminates the architectural debt and provides a solid foundation for optimization workflows.