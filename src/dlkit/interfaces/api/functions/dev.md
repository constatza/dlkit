# Functions Module (Core API)

## Overview
The functions module provides the main public API functions for DLKit workflows. These are the primary entry points for training, inference, and optimization operations, offering a simplified interface that wraps the command/service architecture.

## Architecture & Design Patterns
- **Facade Pattern**: Simple function interface hides command/service complexity
- **Delegation**: Functions delegate to command dispatcher for actual execution
- **Parameter Forwarding**: Functions accept all overrides and forward to commands
- **Type Safety**: Full type hints for all parameters and returns
- **Backward Compatibility**: BREAKING CHANGE markers for API evolution

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `train()` | Function | Execute training workflow | `TrainingResult` |
| `infer()` | Function | Execute standalone inference (BREAKING CHANGE) | `InferenceResult` |
| `predict_with_config()` | Function | Execute prediction with training config | `InferenceResult` |
| `optimize()` | Function | Execute Optuna optimization | `OptimizationResult` |

## Key Components

### Component 1: `train()`

**Purpose**: Execute training workflow with optional parameter overrides.

**Parameters**:
- `settings: BaseSettingsProtocol` - Parsed configuration
- `mlflow: bool = False` - Enable MLflow tracking
- `checkpoint_path: Path | str | None = None` - Resume from checkpoint
- `root_dir: Path | str | None = None` - Override root directory
- `output_dir: Path | str | None = None` - Override output directory
- `data_dir: Path | str | None = None` - Override data directory
- `epochs: int | None = None` - Override training epochs
- `batch_size: int | None = None` - Override batch size
- `learning_rate: float | None = None` - Override learning rate
- `mlflow_host: str | None = None` - Override MLflow server host
- `mlflow_port: int | None = None` - Override MLflow server port
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `**additional_overrides: Any` - Extra parameter overrides

**Returns**: `TrainingResult` with metrics, artifacts, checkpoint path

**Example**:
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

# Load settings
settings = GeneralSettings.from_toml("config.toml")

# Execute training with overrides
result = train(
    settings,
    mlflow=True,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    experiment_name="my_experiment",
)

print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Best checkpoint: {result.checkpoint_path}")
print(f"Metrics: {result.metrics}")
```

### Component 2: `infer()` (BREAKING CHANGE)

**Purpose**: Execute standalone inference from checkpoint only, without requiring training configuration.

**BREAKING CHANGE**: This function now provides checkpoint-only inference. No training configuration files or datasets are needed. For prediction mode with training config, use `predict_with_config()`.

**Parameters**:
- `checkpoint_path: Path | str` - Path to trained model checkpoint
- `inputs: Any` - Input data (tensors, dict, arrays, or file path)
- `device: str = "auto"` - Device specification ("auto", "cpu", "cuda", "mps")
- `batch_size: int = 32` - Batch size for processing
- `apply_transforms: bool = True` - Whether to apply fitted transforms

**Returns**: `InferenceResult` with predictions

**Example**:
```python
from dlkit.interfaces.api import infer
import torch

# New inference API - no config needed
result = infer(
    checkpoint_path="model.ckpt", inputs={"x": torch.randn(32, 10)}, device="auto", batch_size=32
)

predictions = result.predictions
```

**Migration**:
```python
# OLD CODE (deprecated):
from dlkit.tools.config import load_inference_settings

settings = load_inference_settings("config.toml")
result = infer(settings, "model.ckpt")

# NEW CODE:
from dlkit.interfaces.api import infer

result = infer("model.ckpt", your_input_data)

# OR for prediction mode with training config:
from dlkit.interfaces.api import predict_with_config
from dlkit.tools.io import load_settings

settings = load_settings("config.toml")
result = predict_with_config(settings, "model.ckpt")
```

### Component 3: `predict_with_config()`

**Purpose**: Execute Lightning-based prediction using training configuration. Replaces old `infer()` for prediction scenarios with datasets.

**Parameters**:
- `training_settings: TrainingWorkflowSettings` - Training workflow settings
- `checkpoint_path: Path | str` - Path to model checkpoint
- `**overrides: Any` - Additional parameter overrides

**Returns**: `InferenceResult` with predictions

**Example**:
```python
from dlkit.interfaces.api import predict_with_config
from dlkit.tools.io import load_settings

# Load settings
settings = load_settings("config.toml")

# Execute prediction
result = predict_with_config(settings, checkpoint_path="best.ckpt", batch_size=64)

predictions = result.predictions
```

### Component 4: `optimize()`

**Purpose**: Execute Optuna hyperparameter optimization workflow.

**Parameters**:
- `settings: BaseSettingsProtocol` - Configuration with `[OPTUNA].enabled=true`
- `trials: int = 100` - Number of optimization trials
- `mlflow: bool = False` - Enable MLflow tracking
- `checkpoint_path: Path | str | None = None` - Warm-start checkpoint
- `root_dir: Path | str | None = None` - Override root directory
- `output_dir: Path | str | None = None` - Override output directory
- `data_dir: Path | str | None = None` - Override data directory
- `study_name: str | None = None` - Override Optuna study name
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `**additional_overrides: Any` - Extra parameter overrides

**Returns**: `OptimizationResult` with best trial, training result, study summary

**Example**:
```python
from dlkit.interfaces.api import optimize
from dlkit.tools.config import GeneralSettings

# Load settings with Optuna configuration
settings = GeneralSettings.from_toml("optuna_config.toml")

# Execute optimization
result = optimize(settings, trials=50, mlflow=True, study_name="hyperparameter_search")

print(f"Best trial: {result.best_trial}")
print(f"Best parameters: {result.study_summary}")
print(f"Training result: {result.training_result}")
```

## Usage Patterns

### Simple Training
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")
result = train(settings)
```

### Training with Overrides
```python
result = train(settings, mlflow=True, epochs=50, batch_size=64, experiment_name="experiment_1")
```

### Standalone Inference (New)
```python
from dlkit.interfaces.api import infer

# No config needed
result = infer("model.ckpt", input_data)
```

### Prediction with Config
```python
from dlkit.interfaces.api import predict_with_config

result = predict_with_config(training_settings, "model.ckpt")
```

### Hyperparameter Optimization
```python
from dlkit.interfaces.api import optimize

result = optimize(settings, trials=100, mlflow=True)
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
3. **Type checking**: Type hints validated
4. **BREAKING CHANGE handling**: New inference API works correctly

## Related Modules
- `dlkit.interfaces.api.commands`: Functions delegate to command dispatcher
- `dlkit.interfaces.api.domain`: Result types returned by functions
- `dlkit.interfaces.inference.api`: New inference API implementation

## Change Log
- **2025-10-03**: Added comprehensive documentation
- **2024-12-15**: BREAKING CHANGE - New standalone inference API
- **2024-11-01**: Initial core API functions
