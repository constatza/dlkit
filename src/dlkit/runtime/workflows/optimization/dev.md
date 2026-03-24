# Optimization Workflow Module

## Overview
The optimization module provides a complete Domain-Driven Design (DDD) architecture for hyperparameter optimization in DLKit. It implements the Optuna Study → Trial hierarchy with proper separation of concerns across domain, application, and infrastructure layers. The module integrates cleanly with MLflow tracking and supports pluggable storage backends, sampling strategies, and pruning algorithms.

## Architecture & Design Patterns
- **Domain-Driven Design (DDD)**: Clear separation of domain models, application services, and infrastructure adapters
- **Dependency Inversion Principle (DIP)**: All layers depend on abstractions (protocols) not concretions
- **Repository Pattern**: Study persistence abstracted behind `IStudyRepository` interface
- **Adapter Pattern**: MLflow tracking and Optuna storage adapted to domain interfaces
- **Null Object Pattern**: `NullTrackingAdapter`, `NullConfigurationPersister` eliminate conditional logic
- **Service Layer Pattern**: Application services (`StudyManager`, `TrialExecutor`, `OptimizationOrchestrator`) coordinate workflows
- **Factory Pattern**: `OptimizationServiceFactory` creates and wires dependencies
- **Strategy Pattern**: Pluggable samplers, pruners, and persistence formats

Key architectural decisions:
- Pure domain models with no infrastructure dependencies
- Study as aggregate root containing Trials
- Orchestrator manages tracker context lifecycle (enter/exit)
- Configuration persistence optional via interface
- Checkpoint saving disabled for exploratory trials, enabled for best retrain
- MLflow epoch logging injected during trial execution
- Best-effort error handling with comprehensive logging

## Module Structure

### Domain Layer (`domain/`)
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `Study` | Dataclass | Aggregate root for optimization study | N/A |
| `Trial` | Dataclass | Individual optimization trial | N/A |
| `OptimizationResult` | Dataclass | Complete optimization outcome | N/A |
| `OptimizationDirection` | Enum | Minimize or maximize objective | N/A |
| `TrialState` | Enum | Trial execution state | N/A |
| `HyperParameter` | Dataclass | Single hyperparameter definition | N/A |
| `IStudyRepository` | Protocol | Study persistence abstraction | N/A |
| `IExperimentTracker` | Protocol | Experiment tracking abstraction | N/A |
| `IConfigurationPersistence` | Protocol | Config persistence abstraction | N/A |
| `ITrialExecutor` | Protocol | Trial execution abstraction | N/A |

### Application Layer (`application/`)
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `StudyManager` | Service | Study lifecycle management | N/A |
| `TrialExecutor` | Service | Individual trial execution | N/A |
| `OptimizationOrchestrator` | Service | Complete optimization workflow coordination | N/A |

### Infrastructure Layer (`infrastructure/`)
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `OptunaStudyRepository` | Adapter | Optuna-based study persistence | N/A |
| `InMemoryStudyRepository` | Adapter | In-memory study persistence for testing | N/A |
| `MLflowTrackingAdapter` | Adapter | MLflow experiment tracking | N/A |
| `NullTrackingAdapter` | Adapter | No-op tracking when disabled | N/A |
| `TOMLConfigurationPersister` | Adapter | TOML config persistence | N/A |
| `JSONConfigurationPersister` | Adapter | JSON config persistence | N/A |
| `NullConfigurationPersister` | Adapter | No-op persistence when disabled | N/A |

### Factory & Strategy
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `OptimizationServiceFactory` | Factory | Dependency injection and service creation | N/A |
| `OptimizationStrategy` | Strategy | Bridge between IOptimizationStrategy and DDD architecture | N/A |
| `create_optimization_orchestrator` | Function | Factory function for creating orchestrator | `OptimizationOrchestrator` |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.domain`: Result objects (`TrainingResult`, `OptimizationResult`), exceptions (`WorkflowError`)
- `dlkit.tools.config`: Configuration management (`GeneralSettings`)
- `dlkit.tools.io`: Configuration I/O (`write_config`) and locations (`locations.output()`)
- `dlkit.tools.utils.logging_config`: Logger configuration
- `dlkit.runtime.workflows.factories.build_factory`: Component construction
- `dlkit.runtime.workflows.strategies.core`: Core execution (`VanillaExecutor`)
- `dlkit.runtime.workflows.strategies.tracking`: MLflow tracker integration
- `dlkit.core.training.callbacks`: MLflow epoch logger

### External Dependencies
- `optuna`: Hyperparameter optimization framework
- `mlflow`: Experiment tracking platform
- `dataclasses`: Domain model construction
- `contextlib`: Context manager utilities
- `datetime`: Timing information
- `uuid`: Unique ID generation

## Key Components

### Domain Models

#### Component: `Study`

**Purpose**: Aggregate root representing an optimization study containing multiple trials. Models the core Optuna concept of optimizing an objective through trial attempts.

**Properties**:
- `study_id: str` - Unique identifier
- `study_name: str` - Human-readable name
- `direction: OptimizationDirection` - Optimization direction
- `trials: tuple[Trial, ...]` - All trials in study
- `created_at: datetime` - Creation timestamp
- `completed_at: datetime | None` - Completion timestamp
- `target_trials: int` - Number of trials to run
- `pruner_config: dict | None` - Pruner configuration
- `sampler_config: dict | None` - Sampler configuration

**Computed Properties**:
- `is_complete: bool` - Whether study completed target trials
- `successful_trials: list[Trial]` - All successfully completed trials
- `best_trial: Trial | None` - Best trial based on direction
- `best_hyperparameters: dict` - Hyperparameters from best trial
- `best_objective_value: float | None` - Objective value from best trial
- `duration_seconds: float` - Total study duration

**Methods**:
- `add_trial(trial: Trial) -> Study` - Return a new study with the trial appended
- `update_trial(trial_id: str, **updates) -> Study` - Return a new study with the trial updated
- `get_trial(trial_id: str) -> Trial | None` - Retrieve trial by ID
- `complete_study() -> Study` - Return a new study marked as completed

**Example**:
```python
from dlkit.runtime.workflows.optimization import Study, OptimizationDirection

study = Study(
    study_id="study_123",
    study_name="my_optimization",
    direction=OptimizationDirection.MINIMIZE,
    target_trials=100,
    sampler_config={"type": "TPESampler", "params": {"seed": 42}},
)

# Check completion status
if study.is_complete:
    print(f"Best params: {study.best_hyperparameters}")
    print(f"Best value: {study.best_objective_value}")

# Immutable updates return a new aggregate
study = study.add_trial(trial)
study = study.complete_study()

# Access trials
for trial in study.successful_trials:
    print(f"Trial {trial.trial_number}: {trial.objective_value}")
```

---

#### Component: `Trial`

**Purpose**: Domain model representing a single optimization trial with hyperparameters, results, and state information.

**Properties**:
- `trial_id: str` - Unique identifier
- `trial_number: int` - Trial sequence number
- `hyperparameters: dict[str, Any]` - Sampled hyperparameters
- `objective_value: float | None` - Objective function result
- `state: TrialState` - Execution state
- `training_result: TrainingResult | None` - Full training result
- `started_at: datetime | None` - Start timestamp
- `completed_at: datetime | None` - Completion timestamp
- `pruned_at_step: int | None` - Pruning step if pruned

**Computed Properties**:
- `duration_seconds: float` - Trial duration
- `is_complete: bool` - Successfully completed
- `is_failed: bool` - Failed execution
- `is_pruned: bool` - Pruned early

**Example**:
```python
from dlkit.runtime.workflows.optimization import Trial, TrialState

trial = Trial(
    trial_id="trial_001",
    trial_number=1,
    hyperparameters={"learning_rate": 0.001, "batch_size": 32},
    objective_value=0.25,
    state=TrialState.COMPLETE,
)

if trial.is_complete:
    print(f"Trial completed in {trial.duration_seconds}s")
    print(f"Objective: {trial.objective_value}")
```

---

#### Component: `OptimizationResult`

**Purpose**: Domain model for complete optimization results containing study, best trial, and training outcome.

**Properties**:
- `study: Study` - Complete study with all trials
- `best_trial: Trial | None` - Best performing trial
- `best_training_result: TrainingResult | None` - Training result from best retrain
- `total_duration_seconds: float` - Total optimization time

**Computed Properties**:
- `best_hyperparameters: dict` - Best hyperparameters found
- `best_objective_value: float | None` - Best objective value
- `total_trials: int` - Number of trials run
- `successful_trials: int` - Number of successful trials
- `study_summary: dict` - Complete study summary

**Example**:
```python
from dlkit.runtime.workflows.optimization import create_optimization_orchestrator

orchestrator = create_optimization_orchestrator(settings)
result = orchestrator.execute_optimization(
    study_name="hp_search",
    base_settings=settings,
    n_trials=50,
    direction=OptimizationDirection.MINIMIZE,
)

print(f"Best hyperparameters: {result.best_hyperparameters}")
print(f"Successful trials: {result.successful_trials}/{result.total_trials}")
print(f"Total duration: {result.total_duration_seconds}s")
```

---

### Application Services

#### Component: `OptimizationOrchestrator`

**Purpose**: Main orchestrator service coordinating all optimization workflow aspects including study management, trial execution, tracking, and persistence.

**Constructor Parameters**:
- `study_manager: StudyManager` - Study lifecycle management
- `trial_executor: TrialExecutor` - Trial execution service
- `experiment_tracker: IExperimentTracker | None` - Optional tracking
- `config_persister: IConfigurationPersistence | None` - Optional persistence

**Key Methods**:
- `execute_optimization(...) -> OptimizationResult` - Execute complete optimization workflow

**Parameters**:
- `study_name: str` - Study name
- `base_settings: GeneralSettings` - Base configuration
- `n_trials: int` - Number of trials
- `direction: OptimizationDirection` - Minimize/maximize
- `sampler_config: dict | None` - Sampler configuration
- `pruner_config: dict | None` - Pruner configuration
- `storage_config: dict | None` - Storage configuration

**Returns**: `OptimizationResult` - Complete optimization outcome

**Raises**: `WorkflowError` - If optimization fails

**Example**:
```python
from dlkit.runtime.workflows.optimization import (
    OptimizationOrchestrator,
    StudyManager,
    TrialExecutor,
    OptimizationDirection,
)

# Create dependencies
study_manager = StudyManager(repository)
trial_executor = TrialExecutor(build_factory)

# Create orchestrator
orchestrator = OptimizationOrchestrator(
    study_manager=study_manager,
    trial_executor=trial_executor,
    experiment_tracker=mlflow_tracker,
)

# Execute optimization
result = orchestrator.execute_optimization(
    study_name="my_study",
    base_settings=settings,
    n_trials=100,
    direction=OptimizationDirection.MINIMIZE,
)
```

**Implementation Notes**:
- Manages tracker context lifecycle (enters before work, exits after)
- Creates proper nested MLflow run structure (study → trials → best retrain)
- Disables checkpointing for exploratory trials (disk space optimization)
- Enables checkpointing only for best model retrain
- Injects MLflow epoch logger for trial metric logging
- Handles pruned and failed trials gracefully
- Logs trial configuration and results following TrackingDecorator pattern
- TODO: Fix timing calculation (currently placeholder)
- TODO: Implement proper hyperparameter sampling (currently placeholder)

---

#### Component: `TrialExecutor`

**Purpose**: Service responsible for executing individual optimization trials including hyperparameter sampling, training execution, and result collection.

**Constructor Parameters**:
- `build_factory: BuildFactory` - Factory for building training components

**Key Methods**:
- `execute_trial(trial, base_settings, hyperparameters, trial_context, enable_checkpointing) -> TrainingResult`

**Parameters**:
- `trial: Trial` - Trial domain model
- `base_settings: GeneralSettings` - Base configuration
- `hyperparameters: dict` - Hyperparameters for trial
- `trial_context: Any | None` - Optional trial run context for metric logging
- `enable_checkpointing: bool` - Whether to enable checkpointing (default False)

**Returns**: `TrainingResult` - Training outcome

**Raises**:
- `TrialPrunedException` - If trial pruned
- `TrialFailedException` - If trial execution fails

**Example**:
```python
from dlkit.runtime.workflows.optimization import TrialExecutor, Trial

executor = TrialExecutor(build_factory)

trial = Trial(trial_id="trial_1", trial_number=1, hyperparameters={})

# Execute trial without checkpointing (exploratory)
result = executor.execute_trial(
    trial=trial,
    base_settings=base_settings,
    hyperparameters={"learning_rate": 0.001},
    trial_context=None,
    enable_checkpointing=False,
)

# Execute best trial with checkpointing
best_result = executor.execute_trial(
    trial=best_trial,
    base_settings=base_settings,
    hyperparameters=best_hyperparameters,
    trial_context=retrain_context,
    enable_checkpointing=True,  # Save checkpoints for best model
)
```

**Implementation Notes**:
- Uses `VanillaExecutor` for actual training execution
- Applies hyperparameters to base settings via immutable patching
- Disables `ModelCheckpoint` callbacks for exploratory trials
- Injects `MLflowEpochLogger` when trial context provided
- Extracts objective value from common metric keys (val_loss, loss, etc.)
- Handles pruning and failure exceptions appropriately
- TODO: Implement proper hyperparameter application

---

#### Component: `StudyManager`

**Purpose**: Service responsible for study lifecycle management including creation, retrieval, persistence, and completion.

**Constructor Parameters**:
- `repository: IStudyRepository` - Study repository implementation

**Key Methods**:
- `create_study(...) -> Study` - Create new optimization study
- `get_study(study_id: str) -> Study | None` - Retrieve study by ID
- `save_study(study: Study)` - Save study to repository
- `complete_study(study_id: str)` - Mark study as completed

**Example**:
```python
from dlkit.runtime.workflows.optimization import (
    StudyManager,
    OptimizationDirection,
)

manager = StudyManager(repository)

# Create study
study = manager.create_study(
    study_name="my_experiment",
    direction=OptimizationDirection.MINIMIZE,
    target_trials=50,
)

# Complete study
manager.complete_study(study.study_id)

# Retrieve study
retrieved = manager.get_study(study.study_id)
```

**Implementation Notes**:
- Delegates all persistence to repository interface
- Logs study lifecycle events comprehensively
- Rebinds the immutable `Study` aggregate returned by lifecycle operations
- Wraps repository exceptions in `WorkflowError`

---

### Infrastructure Adapters

#### Component: `MLflowTrackingAdapter`

**Purpose**: MLflow adapter implementing proper nested run hierarchy for optimization (Study → Trials → Best Retrain).

**Constructor Parameters**:
- `mlflow_tracker: Any | None` - Existing MLflowTracker instance
- `mlflow_settings: Any | None` - MLflow configuration
- `session_name: str | None` - Session name for experiment

**Context Manager**: Must be used as context manager to manage resource lifecycle

**Key Methods**:
- `create_study_run(study: Study) -> AbstractContextManager[IStudyRunContext]` - Create parent study run
- `create_trial_run(trial: Trial, parent_context) -> AbstractContextManager[ITrialRunContext]` - Create nested trial run
- `create_best_retrain_run(study: Study, parent_context) -> AbstractContextManager[ITrialRunContext]` - Create best retrain run

**Returns**: Context managers for study and trial runs

**Example**:
```python
from dlkit.runtime.workflows.optimization import MLflowTrackingAdapter

adapter = MLflowTrackingAdapter(
    mlflow_settings=settings.MLFLOW,
    session_name="optimization_experiment",
)

# Adapter manages tracker lifecycle
with adapter:
    # Create study run (parent)
    with adapter.create_study_run(study) as study_context:
        study_context.log_study_metadata(study)

        # Create trial runs (nested)
        for trial in trials:
            with adapter.create_trial_run(trial, study_context) as trial_context:
                trial_context.log_trial_hyperparameters(trial.hyperparameters)
                trial_context.log_trial_metrics(metrics)

        # Create best retrain run (nested)
        with adapter.create_best_retrain_run(study, study_context) as retrain_context:
            retrain_context.log_trial_settings(best_settings)
```

**Implementation Notes**:
- Uses `ExitStack` for nested context management
- Delegates to existing `MLflowTracker` for server management
- Creates proper MLflow hierarchy (parent study, nested trials)
- Experiment name from session name or "DLKit_Optimization" default
- Logs comprehensive metadata, metrics, and artifacts
- Saves TOML configurations as artifacts
- Handles non-numeric metrics gracefully
- Null tracker available for tracking-disabled scenarios

---

#### Component: `OptunaStudyRepository`

**Purpose**: Repository implementation using Optuna as persistence layer, translating between domain models and Optuna objects.

**Constructor Parameters**:
- `optuna_module: Any | None` - Optuna module for DI/testing

**Key Methods**:
- `create_study(...) -> Study` - Create study using Optuna
- `get_study(study_id: str) -> Study | None` - Retrieve study from Optuna
- `save_study(study: Study)` - Save study (no-op for Optuna)
- `add_trial_to_study(study_id, trial)` - Add trial to study
- `update_trial_in_study(study_id, trial_id, **updates)` - Update trial
- `get_best_trial(study_id) -> Trial | None` - Get best trial

**Example**:
```python
from dlkit.runtime.workflows.optimization import (
    OptunaStudyRepository,
    OptimizationDirection,
)

repository = OptunaStudyRepository()

# Create study with Optuna backend
study = repository.create_study(
    study_name="my_study",
    direction=OptimizationDirection.MINIMIZE,
    target_trials=100,
    sampler_config={"type": "TPESampler", "params": {"seed": 42}},
    storage_config={"url": "sqlite:///optuna_studies.db"},
)

# Retrieve best trial
best_trial = repository.get_best_trial(study.study_id)
```

**Implementation Notes**:
- Maps domain directions to Optuna directions (MINIMIZE/MAXIMIZE)
- Maps Optuna trial states to domain states
- Builds sampler and pruner from configuration
- Supports storage URL for study persistence
- Generates domain-specific UUIDs for studies
- Maintains internal mapping between domain IDs and Optuna studies
- Converts Optuna trials to domain trials with timing info
- Extracts pruning step from intermediate values

---

#### Component: `TOMLConfigurationPersister`

**Purpose**: TOML configuration persistence implementation for saving best optimization configurations.

**Key Methods**:
- `save_best_configuration(study: Study, configuration: dict | GeneralSettings) -> str | None`

**Returns**: Path to saved configuration file if successful, None otherwise

**Example**:
```python
from dlkit.runtime.workflows.optimization import TOMLConfigurationPersister

persister = TOMLConfigurationPersister()

# Save best configuration
config_path = persister.save_best_configuration(
    study=study,
    configuration=best_settings,
)
# File saved to: output/optuna_results/best_config_study_my_study_trial_42.toml
```

**Implementation Notes**:
- Uses `locations.output("optuna_results")` for output directory
- File naming: `best_config_study_{study_name}_trial_{trial_number}.toml`
- Converts dict to `GeneralSettings` if needed
- Writes with `write_config()` using existing infrastructure
- Returns None on failure (fail-safe)
- Null persister available when persistence disabled

---

### Factory & Strategy

#### Component: `OptimizationServiceFactory`

**Purpose**: Factory for creating optimization services with proper dependency injection following SOLID principles.

**Constructor Parameters**:
- `build_factory: BuildFactory | None` - Training component factory
- `study_repository: IStudyRepository | None` - Repository override
- `experiment_tracker: IExperimentTracker | None` - Tracker override
- `config_persister: IConfigurationPersistence | None` - Persister override

**Key Methods**:
- `create_optimization_orchestrator(settings) -> OptimizationOrchestrator` - Create complete orchestrator
- `create_optimization_strategy(settings) -> OptimizationStrategy` - Create strategy implementation
- `extract_optimization_config(settings) -> dict` - Extract config from settings

**Returns**: Configured services with all dependencies wired

**Example**:
```python
from dlkit.runtime.workflows.optimization import OptimizationServiceFactory

# Create factory with defaults
factory = OptimizationServiceFactory()

# Create orchestrator with dependency injection
orchestrator = factory.create_optimization_orchestrator(settings)

# Create strategy for IOptimizationStrategy compatibility
strategy = factory.create_optimization_strategy(settings)

# Extract configuration
config = factory.extract_optimization_config(settings)
print(f"Trials: {config['n_trials']}")
print(f"Direction: {config['direction']}")
```

**Implementation Notes**:
- Dependency overrides for testing via constructor
- Creates OptunaStudyRepository if enabled, else InMemoryStudyRepository
- Creates MLflowTrackingAdapter if enabled, else NullTrackingAdapter
- Creates TOMLConfigurationPersister if enabled, else NullConfigurationPersister
- Falls back gracefully on failures
- Extracts sampler/pruner/storage config from settings
- Injects SESSION.seed into sampler if not specified
- Study name from OPTUNA.study_name or SESSION.name fallback

---

#### Component: `OptimizationStrategy`

**Purpose**: Bridge between `IOptimizationStrategy` interface and new DDD architecture.

**Constructor Parameters**:
- `factory: OptimizationServiceFactory` - Service factory
- `settings: GeneralSettings` - Configuration settings

**Key Methods**:
- `execute_optimization(settings: GeneralSettings) -> APIOptimizationResult` - Execute optimization

**Returns**: API-compatible optimization result

**Example**:
```python
from dlkit.runtime.workflows.optimization import (
    OptimizationServiceFactory,
    OptimizationStrategy,
)

factory = OptimizationServiceFactory()
strategy = factory.create_optimization_strategy(settings)

# Implements IOptimizationStrategy interface
result = strategy.execute_optimization(settings)

# Returns API-compatible result
print(f"Best trial: {result.best_trial.number}")
print(f"Best value: {result.best_trial.value}")
print(f"Training result: {result.training_result}")
```

**Implementation Notes**:
- Creates orchestrator via factory
- Extracts configuration via factory
- Converts domain result to API result
- Provides `_APICompatibleTrial` wrapper for backward compatibility
- Calculates total duration from start time
- Orchestrator manages tracker context (strategy doesn't enter/exit)

## Usage Patterns

### Common Use Case 1: Basic Optimization Workflow
```python
from dlkit.runtime.workflows.optimization import (
    create_optimization_orchestrator,
    OptimizationDirection,
)
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("config.toml")

# Create orchestrator
orchestrator = create_optimization_orchestrator(settings)

# Execute optimization
result = orchestrator.execute_optimization(
    study_name="hyperparameter_search",
    base_settings=settings,
    n_trials=100,
    direction=OptimizationDirection.MINIMIZE,
)

# Access results
print(f"Best hyperparameters: {result.best_hyperparameters}")
print(f"Best objective: {result.best_objective_value}")
print(f"Successful trials: {result.successful_trials}/{result.total_trials}")

# Best training result available
if result.best_training_result:
    print(f"Best model metrics: {result.best_training_result.metrics}")
    print(f"Checkpoints: {result.best_training_result.artifacts}")
```

### Common Use Case 2: Optimization with MLflow Tracking
```python
from dlkit.runtime.workflows.optimization import OptimizationServiceFactory

# Configuration with MLflow enabled
settings = GeneralSettings.from_toml("config_with_mlflow.toml")

# Factory automatically creates MLflow tracker
factory = OptimizationServiceFactory()
orchestrator = factory.create_optimization_orchestrator(settings)

# Orchestrator manages tracker lifecycle
result = orchestrator.execute_optimization(
    study_name="tracked_optimization",
    base_settings=settings,
    n_trials=50,
    direction=OptimizationDirection.MINIMIZE,
)

# Check MLflow UI for nested run structure:
# - Study run (parent)
#   - Trial 0 run (child)
#   - Trial 1 run (child)
#   - ...
#   - Best retrain run (child)
```

### Common Use Case 3: Custom Repository and Persistence
```python
from dlkit.runtime.workflows.optimization import (
    OptimizationServiceFactory,
    InMemoryStudyRepository,
    JSONConfigurationPersister,
)

# Create custom dependencies
custom_repository = InMemoryStudyRepository()
custom_persister = JSONConfigurationPersister()

# Inject dependencies via factory
factory = OptimizationServiceFactory(
    study_repository=custom_repository,
    config_persister=custom_persister,
)

orchestrator = factory.create_optimization_orchestrator(settings)
result = orchestrator.execute_optimization(...)

# Best config saved as JSON instead of TOML
```

### Common Use Case 4: Using as IOptimizationStrategy
```python
from dlkit.runtime.workflows.optimization import OptimizationServiceFactory
from dlkit.runtime.workflows.strategies.core import IOptimizationStrategy

# Create strategy implementing IOptimizationStrategy
factory = OptimizationServiceFactory()
strategy: IOptimizationStrategy = factory.create_optimization_strategy(settings)

# Use in existing workflows expecting IOptimizationStrategy
result = strategy.execute_optimization(settings)

# Result compatible with API expectations
print(f"Best trial number: {result.best_trial.number}")
print(f"Training metrics: {result.training_result.metrics}")
```

### Common Use Case 5: Study Persistence with Storage
```python
config_toml = """
[OPTUNA]
enabled = true
study_name = "persistent_study"
n_trials = 100
storage = "sqlite:///my_studies.db"

[OPTUNA.sampler]
factory = "optuna.samplers.TPESampler"
seed = 42
"""

settings = GeneralSettings.from_toml_string(config_toml)

# First optimization run
orchestrator = create_optimization_orchestrator(settings)
result1 = orchestrator.execute_optimization(
    study_name="persistent_study",
    base_settings=settings,
    n_trials=50,
    direction=OptimizationDirection.MINIMIZE,
)

# Resume with more trials (study loaded from storage)
result2 = orchestrator.execute_optimization(
    study_name="persistent_study",  # Same name
    base_settings=settings,
    n_trials=50,  # Additional trials
    direction=OptimizationDirection.MINIMIZE,
)
# Total: 100 trials across both runs
```

## Error Handling

**Exceptions Raised**:
- `WorkflowError`: Study/trial creation fails, optimization execution fails
- `TrialPrunedException`: Trial pruned (handled gracefully by orchestrator)
- `TrialFailedException`: Trial execution fails (handled gracefully)
- `ValueError`: Trial ID already exists, trial not found
- Various Optuna exceptions propagate from repository

**Error Handling Pattern**:
```python
from dlkit.runtime.workflows.optimization import create_optimization_orchestrator
from dlkit.interfaces.api.domain import WorkflowError
from dlkit.runtime.workflows.optimization.domain import TrialPrunedException

try:
    orchestrator = create_optimization_orchestrator(settings)
    result = orchestrator.execute_optimization(
        study_name="my_study",
        base_settings=settings,
        n_trials=100,
        direction=OptimizationDirection.MINIMIZE,
    )
except WorkflowError as e:
    logger.error(f"Optimization failed: {e}")
    logger.debug(f"Context: {e.context}")
    # Handle configuration or execution errors
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle unexpected failures
```

**Fail-Safe Design**:
- Tracking failures don't crash optimization
- Persistence failures logged but don't interrupt
- Pruned and failed trials handled gracefully
- Configuration extraction has fallback values
- Null implementations for disabled features

## Testing

### Test Coverage
- Unit tests: `tests/runtime/workflows/optimization/`
- Integration tests: `tests/integration/test_optuna_mlflow_integration.py`
- Strategy tests: `tests/runtime/workflows/strategies/test_solid_integration.py`

### Key Test Scenarios
1. **Complete optimization workflow**: Study creation through best retrain
2. **Nested MLflow runs**: Proper hierarchy verification
3. **Trial pruning and failure**: Graceful handling
4. **Configuration persistence**: TOML and JSON formats
5. **Repository implementations**: Optuna and in-memory
6. **Factory dependency injection**: Custom dependencies
7. **Strategy interface compatibility**: IOptimizationStrategy implementation
8. **Checkpoint management**: Disabled for trials, enabled for best

### Fixtures Used
- `general_settings` (from `conftest.py`): Configuration with OPTUNA enabled
- `optimization_factory` (test-specific): Service factory for testing
- `tmp_path` (pytest built-in): Temporary paths for configs

## Performance Considerations
- Checkpoint saving disabled for exploratory trials (disk space)
- Best model only checkpointed during final retrain
- Study persistence via storage backend (SQLite, etc.)
- In-memory repository for testing/development
- MLflow resource lifecycle managed by orchestrator
- Graceful degradation when tracking disabled
- Lazy service creation via factory

## Future Improvements / TODOs
- [ ] Implement actual hyperparameter sampling (currently placeholder)
- [ ] Fix timing calculation in orchestrator (currently placeholder)
- [ ] Add pruning callback injection to trials
- [ ] Support for multi-objective optimization
- [ ] Distributed optimization across workers
- [ ] Custom objective functions beyond val_loss
- [ ] Intermediate value logging for pruning
- [ ] Study resumption and incremental trials
- [ ] Advanced sampler strategies (CMA-ES, etc.)
- [ ] Visualization integration (Optuna dashboard)

## Related Modules
- `dlkit.runtime.workflows.strategies.core`: Core execution strategies used in trials
- `dlkit.runtime.workflows.strategies.tracking`: MLflow tracker integration
- `dlkit.runtime.workflows.strategies.optuna`: Original Optuna optimizer (legacy)
- `dlkit.runtime.workflows.factories.build_factory`: Component construction
- `dlkit.tools.config`: Configuration management
- `dlkit.core.training.callbacks`: MLflow epoch logger

## Change Log
- **2024-10-03**: Complete DDD architecture implementation
- **2024-10-02**: Added MLflow tracking integration with nested runs
- **2024-10-01**: Implemented study and trial domain models
- **2024-09-30**: Created repository pattern for study persistence
- **2024-09-28**: Added configuration persistence adapters
- **2024-09-25**: Integrated with OptimizationStrategy interface
