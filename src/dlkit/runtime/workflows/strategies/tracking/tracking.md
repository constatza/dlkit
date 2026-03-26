# Tracking Strategy Module

## Overview
The tracking module provides a composable experiment tracking layer for DLKit, implementing MLflow integration following SOLID principles. It enables experiment tracking, metric logging, and artifact management through clean abstractions that decouple tracking implementation from core workflow execution.

## Architecture & Design Patterns
- **Dependency Inversion Principle (DIP)**: Abstract interfaces (`IExperimentTracker`, `IRunContext`) separate tracking protocol from MLflow implementation
- **Decorator Pattern**: `TrackingDecorator` adds tracking capabilities to any `ITrainingExecutor` without modifying it
- **Null Object Pattern**: `NullTracker` and `NullRunContext` eliminate conditional logic when tracking is disabled
- **Resource Manager Pattern**: `MLflowResourceManager` ensures guaranteed cleanup of MLflow resources
- **Context Manager Protocol**: Proper lifecycle management using `__enter__` and `__exit__`
- **Factory Pattern**: `MLflowClientFactory` handles client creation with proper configuration

Key architectural decisions:
- Tracking is orthogonal to execution - can be added/removed via composition
- No global state pollution - explicit resource management
- Fail-safe design - tracking failures don't crash workflows
- Protocol-based design enables testing and alternative implementations

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `IExperimentTracker` | Protocol | Abstract experiment tracker interface | N/A |
| `IRunContext` | Protocol | Context for active tracking run | N/A |
| `MLflowTracker` | Class | MLflow implementation of experiment tracker | N/A |
| `TrackingDecorator` | Class | Decorator adding tracking to executors | N/A |
| `NullTracker` | Class | No-op tracker when tracking disabled | N/A |
| `NullRunContext` | Class | No-op run context when tracking disabled | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `MLflowResourceManager` | Class | Centralized MLflow resource lifecycle management | N/A |
| `MLflowClientFactory` | Class | Factory for creating MLflow clients | `MlflowClient` |
| `ClientBasedRunContext` | Class | Run context using MLflow client | N/A |
| `_health_check` | Method | Check MLflow server health status | `dict \| None` |
| `_derive_server_url` | Method | Extract server URL from config | `str \| None` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `IExperimentTracker` | `create_run()`, `log_settings()`, `log_model_parameters()` | Abstract tracking interface |
| `IRunContext` | `run_id`, `log_metrics()`, `log_params()`, `log_artifact()`, `set_tag()` | Active run context interface |

## Dependencies

### Internal Dependencies
- `dlkit.tools.config`: Settings and configuration (`GeneralSettings`, `MLflowSettings`)
- `dlkit.tools.utils.logging_config`: Logger configuration (`get_logger`)
- `dlkit.tools.io.config`: Configuration I/O (`write_config`)

### External Dependencies
- `mlflow`: Experiment tracking platform
- `contextlib`: Context manager utilities (`contextmanager`, `ExitStack`)
- `pathlib`: Path handling

## Key Components

### Component 1: `IExperimentTracker`

**Purpose**: Abstract protocol defining the contract for experiment tracking systems. Enables dependency inversion - workflow code depends on abstraction, not MLflow specifics.

**Methods**:
- `create_run(experiment_name: str | None, run_name: str | None, nested: bool) -> AbstractContextManager[IRunContext]` - Create tracking run context
- `log_settings(settings: GeneralSettings, run_context: IRunContext) -> None` - Log configuration to run
- `log_model_parameters(model: Any, run_context: IRunContext, settings: GeneralSettings) -> None` - Log model hyperparameters

**Example**:
```python
# Using the abstraction - works with MLflowTracker or NullTracker
tracker: IExperimentTracker = MLflowTracker()

with tracker.create_run(experiment_name="my_experiment") as run_context:
    run_context.log_metrics({"loss": 0.5}, step=1)
    run_context.log_params({"learning_rate": 0.001})
```

**Implementation Notes**:
- Follows Interface Segregation Principle - minimal, focused interface
- Return type uses `AbstractContextManager` for type safety
- All implementations must be context manager compatible

---

### Component 2: `MLflowTracker`

**Purpose**: Concrete MLflow implementation of `IExperimentTracker` using resource manager pattern for proper lifecycle management.

**Constructor Parameters**:
- `disable_autostart: bool = False` - Skip automatic server startup
- `skip_health_checks: bool = False` - Skip health validation checks

**Key Methods**:
- `setup_mlflow_config(mlflow_config: Any) -> tuple[str | None, dict | None]` - Store configuration for context entry
- `create_run(...) -> AbstractContextManager[IRunContext]` - Create MLflow run via resource manager
- `log_settings(settings, run_context)` - Save settings as TOML artifact
- `log_model_parameters(model, run_context, settings)` - Extract and log hyperparameters from `settings.MODEL`
- `cleanup_server()` - Manual cleanup (prefer context manager)
- `get_server_url() -> str | None` - Get active server URL
- `get_server_status(server_url) -> dict | None` - Get server health status

**Returns**: Varies by method

**Raises**:
- `RuntimeError`: If MLflow not configured when creating run
- `RuntimeError`: If settings logging fails
- Various MLflow exceptions during resource initialization

**Example**:
```python
from dlkit.runtime.workflows.strategies.tracking import MLflowTracker

tracker = MLflowTracker()
tracker.setup_mlflow_config(mlflow_settings)

# Context manager handles all resource lifecycle
with tracker:
    with tracker.create_run(experiment_name="training") as run:
        run.log_metrics({"accuracy": 0.95}, step=10)
        run.log_params({"batch_size": 32})
        tracker.log_settings(settings, run)
```

**Implementation Notes**:
- Uses `ExitStack` for nested context management
- Resource initialization deferred to `__enter__()` for proper protocol
- `setup_mlflow_config()` only stores config - resources created on context entry
- Health checks skipped after server startup to avoid interference
- Server status cached to minimize redundant health checks
- Cleanup via `ExitStack` ensures proper resource release

---

### Component 3: `MLflowResourceManager`

**Purpose**: Centralized manager for all MLflow resource lifecycle - clients, servers, runs, experiments. Ensures guaranteed cleanup following Resource Manager Pattern.

**Constructor Parameters**:
- `mlflow_config: MLflowSettings | None` - MLflow configuration settings

**Key Attributes**:
- `_state: MLflowResourceState` - Mutable dataclass container for managed resources
- `_config: MLflowSettings` - Configuration settings
- `_is_initialized: bool` - Initialization flag

**Key Methods**:
- `__enter__() -> MLflowResourceManager` - Initialize all resources
- `__exit__(exc_type, exc_val, exc_tb)` - Cleanup all resources
- `create_run(experiment_name, run_name, nested) -> AbstractContextManager[IRunContext]` - Create run context
- `get_server_info() -> Any` - Get server information
- `cleanup()` - Manual cleanup

**Returns**: Self on context entry, run context from `create_run()`

**Example**:
```python
from dlkit.runtime.workflows.strategies.tracking.mlflow_resource_manager import (
    MLflowResourceManager,
)

resource_manager = MLflowResourceManager(mlflow_config)

with resource_manager as rm:
    # Resources initialized: client, server, experiment
    with rm.create_run(experiment_name="exp") as run_context:
        run_context.log_metrics({"loss": 0.3})
# Automatic cleanup on exit
```

**Implementation Notes**:
- Coordinates global MLflow state for Lightning compatibility
- Sets `mlflow.set_tracking_uri()` during initialization for global sync
- Manages nested run stack for hierarchical tracking
- Uses cleanup callbacks for extensible resource release
- Handles server startup errors gracefully
- Prevents double initialization with `_is_initialized` flag

---

### Component 4: `TrackingDecorator`

**Purpose**: Decorator that adds experiment tracking capabilities to any `ITrainingExecutor` without modifying the executor. Implements Open/Closed Principle - open for extension, closed for modification.

**Constructor Parameters**:
- `executor: ITrainingExecutor` - Base executor to decorate
- `tracker: IExperimentTracker` - Tracker implementation to use
- `settings: GeneralSettings` - Configuration settings
- `experiment_name: str | None = None` - Experiment name override
- `run_name: str | None = None` - Run name override

**Key Methods**:
- `execute(model, datamodule, callbacks) -> TrainingResult` - Execute training with tracking
- `_log_initial_metadata(run_context)` - Log settings and model parameters
- `_log_final_metrics(run_context, result)` - Log training results

**Returns**: `TrainingResult` from wrapped executor

**Raises**:
- Propagates exceptions from wrapped executor
- Logs tracking failures but doesn't crash workflow

**Example**:
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.strategies.tracking import TrackingDecorator, MLflowTracker

# Compose tracking onto vanilla execution
base_executor = VanillaExecutor(settings)
tracker = MLflowTracker()
tracked_executor = TrackingDecorator(
    executor=base_executor, tracker=tracker, settings=settings, experiment_name="my_experiment"
)

# Execute with automatic tracking
result = tracked_executor.execute(model, datamodule, callbacks)
```

**Implementation Notes**:
- Wraps executor execution in MLflow run context
- Logs metadata before training, metrics after
- Tracking failures logged but don't interrupt execution (fail-safe)
- Uses Lightning callbacks for epoch-level metrics
- Automatically logs final training result
- Clean separation: tracking orthogonal to execution logic

---

### Component 5: `IRunContext`

**Purpose**: Protocol defining the interface for an active tracking run. Provides methods for logging metrics, parameters, artifacts, and tags.

**Properties**:
- `run_id: str` - Unique identifier for the tracking run

**Methods**:
- `log_metrics(metrics: dict[str, float], step: int | None = None) -> None` - Log metrics
- `log_params(params: dict[str, Any]) -> None` - Log parameters
- `log_artifact(artifact_path: Path, artifact_dir: str = "") -> None` - Log artifact file
- `set_tag(key: str, value: str) -> None` - Set tag on run

**Example**:
```python
# Type-safe run context usage
def train_with_tracking(run_context: IRunContext):
    run_context.log_params({"lr": 0.001, "batch_size": 32})

    for epoch in range(10):
        loss = train_epoch()
        run_context.log_metrics({"loss": loss}, step=epoch)

    run_context.log_artifact(Path("model.pth"))
    run_context.set_tag("status", "completed")
```

**Implementation Notes**:
- Concrete implementations: `ClientBasedRunContext` (MLflow), `NullRunContext` (no-op)
- All methods designed for fail-safe operation
- Step parameter enables time-series metrics

---

### Component 6: `NullTracker` and `NullRunContext`

**Purpose**: Null Object Pattern implementations that provide safe no-op behavior when tracking is disabled. Eliminates conditional logic throughout codebase.

**NullTracker Methods**:
- `__enter__() -> NullTracker` - No-op context entry
- `__exit__(...)` - No-op context exit
- `create_run(...) -> AbstractContextManager[IRunContext]` - Returns context yielding `NullRunContext`
- `log_settings(...)` - No-op
- `log_model_parameters(...)` - No-op

**NullRunContext Methods**:
- `run_id -> str` - Returns `"null-run-id"`
- `log_metrics(...)` - No-op
- `log_params(...)` - No-op
- `log_artifact(...)` - No-op
- `set_tag(...)` - No-op

**Example**:
```python
# Same code works with real tracker or null tracker
def execute_workflow(tracker: IExperimentTracker):
    with tracker.create_run("experiment") as run:
        run.log_metrics({"loss": 0.5})
        run.log_params({"lr": 0.001})


# With MLflow tracking
execute_workflow(MLflowTracker())

# Without tracking - no conditionals needed!
execute_workflow(NullTracker())
```

**Implementation Notes**:
- Follows same interface as real implementations
- Zero overhead - all operations are pass statements
- Enables "tracking-optional" code without if/else checks
- Simplifies testing by eliminating tracking dependencies

## Usage Patterns

### Common Use Case 1: Basic Experiment Tracking
```python
from dlkit.runtime.workflows.strategies.tracking import MLflowTracker
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("config.toml")

# Initialize tracker with config
tracker = MLflowTracker()
tracker.setup_mlflow_config(settings.MLFLOW)

# Use as context manager for automatic cleanup
with tracker:
    with tracker.create_run(experiment_name="training_exp", run_name="run_001") as run:
        # Log hyperparameters
        run.log_params({"learning_rate": 0.001, "batch_size": 32})

        # Training loop with metrics
        for epoch in range(10):
            loss = train_epoch()
            run.log_metrics({"loss": loss, "epoch": epoch}, step=epoch)

        # Log final artifacts
        run.log_artifact(Path("model.pth"), artifact_dir="models")
        tracker.log_settings(settings, run)
```

### Common Use Case 2: Composing Tracking onto Executor
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.strategies.tracking import TrackingDecorator, MLflowTracker

# Create base executor
executor = VanillaExecutor(settings)

# Wrap with tracking - decorator pattern
tracker = MLflowTracker()
tracked_executor = TrackingDecorator(
    executor=executor, tracker=tracker, settings=settings, experiment_name="my_training"
)

# Execute - tracking happens automatically
result = tracked_executor.execute(model, datamodule, callbacks=[])
print(f"Training completed with best metric: {result.metrics['best_val_loss']}")
```

### Common Use Case 3: Nested Runs for Hyperparameter Search
```python
with tracker:
    # Parent run for overall optimization
    with tracker.create_run(experiment_name="hp_search", run_name="grid_search") as parent_run:
        parent_run.log_params({"search_type": "grid", "n_trials": 10})

        for trial_num, params in enumerate(param_grid):
            # Child run for each trial
            with tracker.create_run(
                experiment_name="hp_search", run_name=f"trial_{trial_num}", nested=True
            ) as trial_run:
                trial_run.log_params(params)
                result = train_with_params(params)
                trial_run.log_metrics({"final_loss": result.loss})
```

### Common Use Case 4: Tracking-Optional Execution
```python
from dlkit.runtime.workflows.strategies.tracking import MLflowTracker, NullTracker

# Select tracker based on configuration
tracker = MLflowTracker() if settings.MLFLOW.enabled else NullTracker()

# Same code path regardless of tracking state
with tracker:
    with tracker.create_run("experiment") as run:
        run.log_params({"param": "value"})
        result = train_model()
        run.log_metrics({"accuracy": result.accuracy})
# No conditional logic needed!
```

## Error Handling

**Exceptions Raised**:
- `RuntimeError`: When `create_run()` called before MLflow configuration
- `RuntimeError`: When settings or model parameter logging fails
- `mlflow.exceptions.MlflowException`: Various MLflow-specific errors during resource initialization
- `Exception`: Generic exceptions during resource cleanup (logged, not re-raised)

**Error Handling Pattern**:
```python
try:
    tracker = MLflowTracker()
    tracker.setup_mlflow_config(mlflow_config)

    with tracker:
        with tracker.create_run("exp") as run:
            run.log_metrics({"loss": 0.5})
except RuntimeError as e:
    logger.error(f"Tracking setup failed: {e}")
    # Fall back to null tracker
    tracker = NullTracker()
    # Continue execution without tracking
```

**Fail-Safe Design**:
- Tracking failures don't crash workflows
- `TrackingDecorator` catches and logs tracking errors
- Cleanup errors logged but don't propagate
- Server startup failures captured in `server_start_error`

## Testing

### Test Coverage
- Unit tests: `tests/runtime/workflows/strategies/test_tracking_decorator.py`
- Integration tests: `tests/integration/test_mlflow_training_integration.py`
- Resource management tests: `tests/runtime/workflows/strategies/test_mlflow_resource_manager.py`

### Key Test Scenarios
1. **Basic tracking flow**: Create run, log metrics, verify persistence
2. **Nested runs**: Parent-child run hierarchy with MLflow
3. **Resource cleanup**: Proper server/client cleanup on exit
4. **Error handling**: Tracking failures don't crash execution
5. **Null tracker behavior**: No-op operations when tracking disabled
6. **Decorator composition**: Tracking added to vanilla executor
7. **Configuration handling**: Server URL derivation, health checks

### Fixtures Used
- `mlflow_settings` (from `conftest.py`): MLflow configuration
- `general_settings` (from `conftest.py`): Complete settings object
- `tmp_path` (pytest built-in): Temporary paths for artifacts
- `mock_server` (test-specific): Mock MLflow server for isolation

## Performance Considerations
- Health checks use 0.2s timeout to avoid blocking
- Server status cached to minimize redundant health checks
- Lazy resource initialization - deferred until context entry
- ExitStack for efficient nested context management
- Settings serialization only when logging (not on every run creation)
- Cleanup callbacks allow async resource release

## Future Improvements / TODOs
- [ ] Support for distributed tracking (multiple workers)
- [ ] Artifact compression before upload
- [ ] Batch metrics logging for reduced network calls
- [ ] Plugin system for alternative tracking backends (Weights & Biases, TensorBoard)
- [ ] Automatic experiment naming from git branch/commit
- [ ] Metric comparison and regression detection
- [ ] Integration with cloud storage (S3, GCS) for artifacts

## Related Modules
- `dlkit.runtime.workflows.strategies.core`: Core execution strategies that tracking decorates
- `dlkit.runtime.workflows.strategies.optuna`: Optimization strategies that use tracking for trial logging
- `dlkit.interfaces.servers`: Server management for MLflow backend
- `dlkit.tools.config.mlflow_settings`: Configuration models for MLflow
- `dlkit.core.training.callbacks.mlflow_epoch_logger`: Callback for epoch-level metric logging

## Change Log
- **2024-10-02**: Migrated to `MLflowResourceManager` for centralized resource management
- **2024-10-01**: Added health check caching and skip logic
- **2024-09-30**: Implemented `TrackingDecorator` following decorator pattern
- **2024-09-24**: Initial tracking abstractions with DIP
