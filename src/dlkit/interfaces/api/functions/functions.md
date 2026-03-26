# Functions Module (Core API)

## Overview
The functions module provides the main public API functions for DLKit workflows. These are the primary entry points for training, optimization, and configuration operations, offering a simplified interface that wraps the command/service architecture.

Inference is **not** handled here — use `load_model()` from `dlkit` (see [`interfaces/inference/README.md`](../../inference/README.md)).

## Architecture & Design Patterns
- **Facade Pattern**: Simple function interface hides command/service complexity
- **Delegation**: Functions delegate to command dispatcher for actual execution
- **Parameter Forwarding**: Functions accept all overrides and forward to commands
- **Type Safety**: Full type hints for all parameters and returns

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `train()` | Function | Execute training workflow | `TrainingResult` |
| `optimize()` | Function | Execute Optuna optimization | `OptimizationResult` |
| `execute()` | Function | Unified workflow routing (train or optimize) | `TrainingResult \| OptimizationResult` |
| `validate_config()` | Function | Validate configuration structure | `bool` |
| `generate_template()` | Function | Generate a configuration template | `str` |
| `validate_template()` | Function | Validate a config template | `bool` |
| `load_logged_model()` | Function | Load MLflow-logged model | `CheckpointPredictor` |
| `load_registered_model()` | Function | Load MLflow-registered model | `CheckpointPredictor` |
| `search_logged_models()` | Function | Search MLflow logged models | `list[LoggedModelRecord]` |
| `search_registered_models()` | Function | Search MLflow registered models | results |

**Inference** (`load_model`, `predict`) lives in `dlkit.interfaces.inference` and is re-exported at the top-level `dlkit` package.

## Key Components

### Component 1: `train()`

**Purpose**: Execute training workflow with optional parameter overrides.

**Parameters**:
- `settings: BaseSettingsProtocol` - Parsed configuration
- `checkpoint_path: Path | str | None = None` - Resume from checkpoint
- `root_dir: Path | str | None = None` - Override root directory
- `epochs: int | None = None` - Override training epochs
- `batch_size: int | None = None` - Override batch size
- `learning_rate: float | None = None` - Override learning rate
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `**additional_overrides: Any` - Extra parameter overrides

Note: MLflow tracking is enabled by including an `[MLFLOW]` section in your configuration; there is no boolean `mlflow=` toggle.

**Returns**: `TrainingResult` with metrics, artifacts, checkpoint path

**Example**:
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")
result = train(settings, epochs=100, batch_size=32, learning_rate=0.001)

print(f"Best checkpoint: {result.checkpoint_path}")
print(f"Metrics: {result.metrics}")
```

### Component 2: `optimize()`

**Purpose**: Execute Optuna hyperparameter optimization workflow.

**Parameters**:
- `settings: BaseSettingsProtocol` - Configuration with `[OPTUNA]` section
- `trials: int = 100` - Number of optimization trials
- `checkpoint_path: Path | str | None = None` - Warm-start checkpoint
- `root_dir: Path | str | None = None` - Override root directory
- `study_name: str | None = None` - Override Optuna study name
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `**additional_overrides: Any` - Extra parameter overrides

**Returns**: `OptimizationResult` with best trial, training result, study summary

**Example**:
```python
from dlkit.interfaces.api import optimize
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("optuna_config.toml")
result = optimize(settings, trials=50, study_name="hyperparameter_search")

print(f"Best trial: {result.best_trial}")
```

### Component 3: `execute()`

**Purpose**: Unified workflow routing — automatically dispatches to `train()` or `optimize()` based on whether `[OPTUNA]` is configured.

**Example**:
```python
from dlkit.interfaces.api import execute
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")
result = execute(settings)  # train or optimize depending on config
```

### Component 4: `validate_config()` / `generate_template()`

**Purpose**: Configuration utilities for validation and template generation.

```python
from dlkit.interfaces.api import validate_config, generate_template

# Validate a loaded settings object
is_valid = validate_config(settings, dry_build=True)

# Generate a TOML template
template = generate_template(kind="training")
```

### Component 5: Model Registry Helpers

Functions for loading models tracked in MLflow:

```python
from dlkit.interfaces.api import load_logged_model, load_registered_model

# Load a logged model artifact
predictor = load_logged_model("runs:/abc123/model")

# Load a registered model by alias
predictor = load_registered_model("MyModel", alias="champion")
```

See [`functions/model_logged.py`](model_logged.py) and [`functions/model_registry.py`](model_registry.py) for full parameter details.

## Usage Patterns

### Training
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")
result = train(settings)
```

### Training with Overrides
```python
result = train(settings, epochs=50, batch_size=64, experiment_name="experiment_1")
```

### Inference
```python
# Inference is NOT in this module — use load_model() from dlkit:
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    predictions = predictor.predict(x=input_tensor)
```

### Hyperparameter Optimization
```python
from dlkit.interfaces.api import optimize

result = optimize(settings, trials=100)
```

## Error Handling

```python
from dlkit.interfaces.api import train
from dlkit.interfaces.api.domain import WorkflowError

try:
    result = train(settings, epochs=100)
except WorkflowError as e:
    print(f"Training failed: {e.message}")
    print(f"Context: {e.context}")
```

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/api/test_core_functions.py`
- Integration tests: `tests/integration/test_api_integration.py`

### Key Test Scenarios
1. **Training execution**: Functions delegate to commands correctly
2. **Parameter forwarding**: Overrides passed through properly
3. **Optimization routing**: `execute()` dispatches correctly

## Related Modules
- `dlkit.interfaces.api.commands`: Functions delegate to command dispatcher
- `dlkit.interfaces.api.domain`: Result types returned by functions
- `dlkit.interfaces.inference.api`: `load_model()` — inference entry point
