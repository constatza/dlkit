# Optuna Hyperparameter Optimization Strategy Module

## Overview
The optuna module provides composable hyperparameter optimization capabilities for DLKit, implementing the Optuna framework integration following Dependency Inversion Principle (DIP). It defines clean abstractions for optimization results and optimizers, enabling pluggable hyperparameter search strategies that are decoupled from concrete Optuna implementation details.

## Architecture & Design Patterns
- **Dependency Inversion Principle (DIP)**: Abstract interfaces (`IHyperparameterOptimizer`, `IOptimizationResult`) decouple optimization logic from Optuna specifics
- **Single Responsibility Principle (SRP)**: Settings sampling delegated to dedicated `SettingsSampler` class
- **Strategy Pattern**: Optimization as pluggable strategy with multiple implementations possible
- **Adapter Pattern**: `OptunaOptimizer` adapts Optuna API to DLKit abstractions
- **Factory Pattern**: Sampler and pruner built via `FactoryProvider` from configuration
- **Null Object Pattern**: Graceful handling of missing configuration values

Key architectural decisions:
- Optimization abstracted from concrete Optuna implementation for testability and future alternatives
- Settings sampling separated into dedicated sampler following SRP
- Seed propagation from session to sampler for reproducibility
- Study persistence via optional storage backend
- Best-effort configuration building with silent fallbacks

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `IHyperparameterOptimizer` | Protocol | Abstract hyperparameter optimizer interface | N/A |
| `IOptimizationResult` | Protocol | Abstract optimization result interface | N/A |
| `OptunaOptimizer` | Class | Optuna implementation of hyperparameter optimizer | N/A |
| `OptunaOptimizationResult` | Class | Optuna implementation of optimization result | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_build_sampler_pruner` | Method | Build Optuna sampler and pruner from config | `tuple[Any \| None, Any \| None]` |
| `_derive_study_name` | Method | Derive study name from configuration | `str \| None` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `IHyperparameterOptimizer` | `optimize()`, `create_sampled_settings()` | Abstract optimizer contract |
| `IOptimizationResult` | `best_params`, `best_value`, `trial_number`, `study_summary` | Abstract result contract |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.domain`: Exception types (`WorkflowError`)
- `dlkit.tools.config`: Configuration management (`GeneralSettings`)
- `dlkit.tools.config.core.context`: Build context for factory instantiation
- `dlkit.tools.config.core.factories`: Factory provider for component creation
- `dlkit.tools.config.samplers.optuna_sampler`: Settings sampling logic

### External Dependencies
- `optuna`: Hyperparameter optimization framework
- `typing`: Type hints and protocols
- `collections.abc`: Callable type

## Key Components

### Component 1: `IHyperparameterOptimizer`

**Purpose**: Abstract protocol defining the contract for hyperparameter optimization strategies. Enables dependency inversion - workflows depend on this abstraction, not Optuna.

**Methods**:
- `optimize(objective: Callable[[Any], float], settings: GeneralSettings, n_trials: int, direction: str = "minimize") -> IOptimizationResult` - Run hyperparameter optimization
- `create_sampled_settings(base_settings: GeneralSettings, trial: Any) -> GeneralSettings` - Create settings with sampled hyperparameters

**Parameters**:
- `objective: Callable[[Any], float]` - Objective function to optimize (takes trial, returns metric)
- `settings: GeneralSettings` - Base configuration for optimization
- `n_trials: int` - Number of optimization trials to run
- `direction: str` - Optimization direction ("minimize" or "maximize")
- `base_settings: GeneralSettings` - Base settings for sampling
- `trial: Any` - Trial object for parameter sampling

**Returns**: `IOptimizationResult` - Optimization result with best parameters and metadata

**Example**:
```python
from dlkit.runtime.workflows.strategies.optuna import IHyperparameterOptimizer, OptunaOptimizer

def my_objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # Train with sampled params and return metric
    result = train_with_params(lr, batch_size)
    return result.loss

# Type-safe optimizer usage
optimizer: IHyperparameterOptimizer = OptunaOptimizer()
result = optimizer.optimize(
    objective=my_objective,
    settings=settings,
    n_trials=50,
    direction="minimize"
)

print(f"Best params: {result.best_params}")
print(f"Best value: {result.best_value}")
```

**Implementation Notes**:
- Interface Segregation - focused on optimization concerns only
- Trial object type is `Any` to support different optimization frameworks
- Settings sampling separated into dedicated method for flexibility

---

### Component 2: `IOptimizationResult`

**Purpose**: Abstract protocol defining the contract for optimization results. Provides unified interface to access best parameters, values, and study metadata regardless of backend.

**Properties**:
- `best_params: dict[str, Any]` - Best hyperparameters found during optimization
- `best_value: float` - Best objective value achieved
- `trial_number: int` - Trial number that achieved best result
- `study_summary: dict[str, Any]` - Summary metadata about optimization study

**Example**:
```python
from dlkit.runtime.workflows.strategies.optuna import IOptimizationResult

def analyze_optimization(result: IOptimizationResult):
    print(f"Best trial: {result.trial_number}")
    print(f"Best value: {result.best_value}")
    print(f"Best parameters:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")
    print(f"Study summary: {result.study_summary}")
```

**Implementation Notes**:
- All properties are abstract - implementations must provide concrete values
- Dictionary returns enable flexible parameter schemas
- Study summary structure varies by implementation

---

### Component 3: `OptunaOptimizer`

**Purpose**: Concrete Optuna implementation of `IHyperparameterOptimizer`. Orchestrates Optuna study creation, execution, and result extraction following DIP.

**Constructor Parameters**: None - stateless optimizer

**Key Methods**:
- `optimize(objective, settings, n_trials, direction) -> IOptimizationResult` - Run Optuna optimization study
- `create_sampled_settings(base_settings, trial) -> GeneralSettings` - Create settings with Optuna-sampled hyperparameters
- `apply_best_params(base_settings, best_params) -> GeneralSettings` - Apply best parameters to settings
- `_build_sampler_pruner(opt_cfg, session_seed) -> tuple[Any | None, Any | None]` - Build Optuna sampler and pruner
- `_derive_study_name(opt_cfg, settings) -> str | None` - Derive study name from config

**Returns**:
- `optimize()`: `OptunaOptimizationResult` containing best trial and study
- `create_sampled_settings()`: `GeneralSettings` with sampled hyperparameters
- `apply_best_params()`: `GeneralSettings` with best parameters applied

**Raises**:
- `WorkflowError`: If Optuna not enabled in configuration when `optimize()` called

**Example**:
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer
from dlkit.tools.config import GeneralSettings

# Load configuration with OPTUNA section enabled
settings = GeneralSettings.from_toml("config.toml")

# Initialize optimizer
optimizer = OptunaOptimizer()

# Define objective function using Optuna trial
def objective(trial):
    # Create sampled settings for this trial
    sampled_settings = optimizer.create_sampled_settings(settings, trial)

    # Train with sampled configuration
    result = train_model(sampled_settings)

    # Return metric to optimize
    return result.metrics["val_loss"]

# Run optimization
opt_result = optimizer.optimize(
    objective=objective,
    settings=settings,
    n_trials=100,
    direction="minimize"
)

# Apply best params to settings for final training
final_settings = optimizer.apply_best_params(settings, opt_result.best_params)
final_result = train_model(final_settings)
```

**Implementation Notes**:
- Checks `settings.OPTUNA.enabled` before running optimization
- Builds sampler and pruner via `FactoryProvider` from configuration
- Injects session seed into sampler if sampler seed not configured (reproducibility)
- Study name derived from `OPTUNA.study_name` or `SESSION.name` fallback
- Optional storage backend for study persistence (`OPTUNA.storage`)
- `load_if_exists=True` enables resuming studies when storage configured
- Settings sampling delegated to `SettingsSampler` following SRP
- `apply_best_params()` uses Pydantic `model_copy()` for immutable updates
- Best-effort configuration building - silent failures with None returns

**Seed Propagation Strategy**:
1. Check if `opt_cfg.sampler.seed` is configured
2. If None and `settings.SESSION.seed` exists, inject session seed
3. Pass seed via `BuildContext.overrides` to factory
4. Enables reproducible optimization runs

**Study Configuration**:
- Study name: `opt_cfg.study_name` → `settings.SESSION.name` → None
- Storage: `opt_cfg.storage` (e.g., SQLite URL for persistence)
- Direction: Passed from caller ("minimize" or "maximize")
- Load existing: Enabled when both name and storage provided

---

### Component 4: `OptunaOptimizationResult`

**Purpose**: Concrete implementation of `IOptimizationResult` wrapping Optuna trial and study objects.

**Constructor Parameters**:
- `optuna_trial: Any` - Optuna best trial object
- `study: Any` - Optuna study object

**Properties**:
- `best_params: dict[str, Any]` - Extracts params from `trial.params` attribute
- `best_value: float` - Extracts value from `trial.value`, converted to float
- `trial_number: int` - Extracts number from `trial.number`, converted to int
- `study_summary: dict[str, Any]` - Summary with trials count, direction, study name

**Example**:
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizationResult

# Result wraps Optuna objects
result = OptunaOptimizationResult(study.best_trial, study)

# Access best parameters
print(f"Learning rate: {result.best_params['learning_rate']}")
print(f"Batch size: {result.best_params['batch_size']}")

# Access metadata
print(f"Best loss: {result.best_value}")
print(f"Achieved in trial: {result.trial_number}")
print(f"Total trials: {result.study_summary['trials']}")
print(f"Direction: {result.study_summary['direction']}")
```

**Implementation Notes**:
- Wraps Optuna objects without copying data
- `best_params` returns dict copy for safety
- Graceful handling if `trial.params` attribute missing (returns empty dict)
- Study summary includes direction enum name, not raw enum
- All numeric values converted to Python primitives (float, int)

## Usage Patterns

### Common Use Case 1: Basic Hyperparameter Optimization
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.factories import BuildFactory
from dlkit.tools.config import GeneralSettings

# Load configuration with OPTUNA enabled
settings = GeneralSettings.from_toml("config.toml")

# Initialize components
optimizer = OptunaOptimizer()
executor = VanillaExecutor()
factory = BuildFactory()

# Define objective function
def objective(trial):
    # Sample hyperparameters for this trial
    sampled_settings = optimizer.create_sampled_settings(settings, trial)

    # Build and train with sampled settings
    components = factory.build_training_components(sampled_settings)
    result = executor.execute(components, sampled_settings)

    # Return metric to minimize
    return result.metrics["val_loss"]

# Run optimization
opt_result = optimizer.optimize(
    objective=objective,
    settings=settings,
    n_trials=50,
    direction="minimize"
)

# Train final model with best params
best_settings = optimizer.apply_best_params(settings, opt_result.best_params)
components = factory.build_training_components(best_settings)
final_result = executor.execute(components, best_settings)
```

### Common Use Case 2: Optimization with MLflow Tracking
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.strategies.tracking import TrackingDecorator, MLflowTracker

# Setup tracking
tracker = MLflowTracker()
tracker.setup_mlflow_config(settings.MLFLOW)

# Wrap executor with tracking
executor = VanillaExecutor()
tracked_executor = TrackingDecorator(
    executor=executor,
    tracker=tracker,
    settings=settings
)

optimizer = OptunaOptimizer()

# Objective with nested run tracking
def objective(trial):
    sampled_settings = optimizer.create_sampled_settings(settings, trial)

    # Each trial gets its own tracked run
    with tracker:
        with tracker.create_run(
            experiment_name="hp_search",
            run_name=f"trial_{trial.number}",
            nested=True
        ) as run:
            # Log trial parameters
            run.log_params(trial.params)

            # Execute with tracking
            components = factory.build_training_components(sampled_settings)
            result = tracked_executor.execute(components, sampled_settings)

            # Log trial result
            run.log_metrics({"trial_loss": result.metrics["val_loss"]})

            return result.metrics["val_loss"]

# Run optimization with tracking
opt_result = optimizer.optimize(objective, settings, n_trials=100)
```

### Common Use Case 3: Resumable Optimization with Storage
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer
from dlkit.tools.config import GeneralSettings

# Configuration with storage backend
config_toml = """
[OPTUNA]
enabled = true
study_name = "my_experiment"
storage = "sqlite:///optuna_studies.db"
n_trials = 100

[OPTUNA.sampler]
factory = "optuna.samplers.TPESampler"
seed = 42

[OPTUNA.pruner]
factory = "optuna.pruners.MedianPruner"
n_startup_trials = 5
"""

settings = GeneralSettings.from_toml_string(config_toml)
optimizer = OptunaOptimizer()

# First run - creates study
result1 = optimizer.optimize(objective, settings, n_trials=50)
print(f"Completed {result1.study_summary['trials']} trials")

# Second run - resumes existing study
result2 = optimizer.optimize(objective, settings, n_trials=50)
print(f"Total trials: {result2.study_summary['trials']}")
# Will show 100 trials (50 + 50)
```

### Common Use Case 4: Custom Sampler and Pruner
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer

# Configuration with custom sampler/pruner
config = """
[OPTUNA]
enabled = true

[OPTUNA.sampler]
factory = "optuna.samplers.CmaEsSampler"
restart_strategy = "ipop"
seed = 123

[OPTUNA.pruner]
factory = "optuna.pruners.HyperbandPruner"
min_resource = 1
max_resource = "auto"
reduction_factor = 3
"""

settings = GeneralSettings.from_toml_string(config)
optimizer = OptunaOptimizer()

# Sampler and pruner built automatically from config
result = optimizer.optimize(objective, settings, n_trials=100)
```

### Common Use Case 5: Maximization Objective
```python
optimizer = OptunaOptimizer()

def accuracy_objective(trial):
    sampled_settings = optimizer.create_sampled_settings(settings, trial)
    result = train_model(sampled_settings)
    # Return accuracy to maximize
    return result.metrics["val_accuracy"]

# Maximize instead of minimize
opt_result = optimizer.optimize(
    objective=accuracy_objective,
    settings=settings,
    n_trials=50,
    direction="maximize"  # Find maximum accuracy
)

print(f"Best accuracy: {opt_result.best_value}")
```

## Error Handling

**Exceptions Raised**:
- `WorkflowError`: When `optimize()` called but `OPTUNA.enabled` is False
- Optuna exceptions propagate from `study.optimize()` (e.g., `OptunaError`)

**Error Handling Pattern**:
```python
from dlkit.runtime.workflows.strategies.optuna import OptunaOptimizer
from dlkit.interfaces.api.domain import WorkflowError
import optuna.exceptions

try:
    optimizer = OptunaOptimizer()
    result = optimizer.optimize(objective, settings, n_trials=100)
except WorkflowError as e:
    logger.error(f"Optimization configuration error: {e}")
    # Handle missing or invalid config
except optuna.exceptions.OptunaError as e:
    logger.error(f"Optuna execution error: {e}")
    # Handle Optuna-specific failures
```

**Fail-Safe Design**:
- Sampler/pruner building failures return None (Optuna uses defaults)
- Study name derivation failure returns None (Optuna auto-generates)
- `apply_best_params()` failures return original settings unchanged
- Silent fallbacks for missing configuration attributes

## Testing

### Test Coverage
- Unit tests: `tests/runtime/workflows/strategies/test_optuna_optimizer.py`
- Integration tests: `tests/integration/test_optuna_mlflow_integration.py`
- Strategy tests: `tests/runtime/workflows/strategies/test_solid_integration.py`

### Key Test Scenarios
1. **Basic optimization**: Run optimization, verify best params found
2. **Settings sampling**: Trial sampling produces valid settings
3. **Best params application**: Best parameters applied to settings correctly
4. **Seed propagation**: Session seed injected when sampler seed not configured
5. **Study persistence**: Studies resume when storage configured
6. **Sampler/pruner building**: Factory creates components from config
7. **Error handling**: WorkflowError when optimization not enabled
8. **Direction handling**: Minimize and maximize both work correctly

### Fixtures Used
- `general_settings` (from `conftest.py`): Complete configuration with OPTUNA section
- `optuna_settings` (from `conftest.py`): Optuna-specific configuration
- `tmp_path` (pytest built-in): Temporary paths for SQLite storage

## Performance Considerations
- Study persistence enables incremental optimization across runs
- Sampler/pruner built once per study, not per trial
- Settings sampling creates new settings objects (immutable pattern)
- Best params application uses shallow copy via `model_copy()`
- No global state - optimizer instances are stateless and thread-safe

## Future Improvements / TODOs
- [ ] Support for multi-objective optimization (Pareto fronts)
- [ ] Distributed optimization across multiple workers
- [ ] Custom trial callbacks for intermediate metric logging
- [ ] Visualization integration (Optuna dashboard, plots)
- [ ] Early stopping based on convergence criteria
- [ ] Warmstart from previous study results
- [ ] Configuration validation for OPTUNA section
- [ ] Built-in search spaces for common hyperparameters
- [ ] Integration with neural architecture search (NAS)

## Related Modules
- `dlkit.runtime.workflows.strategies.core`: Core execution strategies used in objective functions
- `dlkit.runtime.workflows.strategies.tracking`: Tracking decorators for trial logging
- `dlkit.tools.config.samplers.optuna_sampler`: Settings sampling implementation
- `dlkit.tools.config.core.factories`: Factory provider for sampler/pruner creation
- `dlkit.runtime.workflows.optimization`: Higher-level optimization orchestration

## Change Log
- **2024-10-03**: Initial Optuna optimizer implementation with DIP
- **2024-10-02**: Added seed propagation from session to sampler
- **2024-10-01**: Delegated settings sampling to SettingsSampler (SRP)
- **2024-09-30**: Added `apply_best_params()` method
- **2024-09-28**: Implemented study persistence with storage backend
- **2024-09-25**: Added sampler/pruner factory building from config
