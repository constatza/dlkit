# Overrides Module

## Overview
The overrides module provides runtime parameter override management for DLKit API, enabling users to modify configuration without editing TOML files. Implements thread-local path context for clean API overrides and maintains immutability through `BasicSettings.patch()`.

## Architecture & Design Patterns
- **Immutability**: Settings never mutated - always copied with overrides
- **Thread-Local Storage**: Path overrides stored per-thread to avoid global pollution
- **Context Manager Protocol**: Path overrides applied within context scope
- **Separation of Concerns**: Path overrides separate from settings overrides
- **Type Safety**: All overrides validated and type-checked
- **Settings API Integration**: Uses `BasicSettings.patch()` for type-safe updates

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `BasicOverrideManager` | Class | Apply runtime overrides to settings | `GeneralSettings` |
| `PathOverrideContext` | Dataclass | Context for API path overrides | N/A |
| `get_current_path_context()` | Function | Get current path context | `PathOverrideContext \| None` |
| `set_path_context()` | Function | Set path context for thread | `None` |
| `path_override_context()` | Context Manager | Apply path overrides in scope | `None` |
| `resolve_with_context()` | Function | Resolve path with override context | `Path` |
| `basic_override_manager` | Instance | Global override manager | `BasicOverrideManager` |

## Key Components

### Component 1: `BasicOverrideManager`

**Purpose**: Apply runtime parameter overrides to GeneralSettings while maintaining immutability.

**Key Methods**:
- `apply_overrides(base_settings: GeneralSettings, **overrides: Any) -> GeneralSettings`
- `validate_overrides(settings: GeneralSettings, **overrides: Any) -> list[str]`

**Override Categories**:
- **Path overrides**: `root_dir`, `output_dir`, `data_dir`, `checkpoints_dir`
- **Model overrides**: `checkpoint_path`
- **Training overrides**: `epochs`, `batch_size`, `learning_rate`
- **MLflow overrides**: `mlflow_host`, `mlflow_port`, `experiment_name`, `run_name`
- **Optuna overrides**: `trials`, `study_name`

**Example**:
```python
from dlkit.interfaces.api.overrides import basic_override_manager
from pathlib import Path

# Apply overrides
new_settings = basic_override_manager.apply_overrides(
    settings,
    checkpoint_path=Path("./model.ckpt"),
    epochs=100,
    batch_size=32,
    mlflow_host="localhost"
)

# Validate before applying
errors = basic_override_manager.validate_overrides(
    settings,
    epochs=-10,  # Invalid
    checkpoint_path=Path("./missing.ckpt")  # Doesn't exist
)
if errors:
    print(f"Validation errors: {errors}")
```

**Implementation Notes**:
- Uses `BasicSettings.patch()` for immutable updates
- Path overrides set thread-local context via `set_path_context()`
- Training overrides update both top-level and nested fields for consistency
- Validates numeric parameters are positive
- Validates checkpoint files exist
- Validates plugin overrides require corresponding plugin enabled

### Component 2: `PathOverrideContext`

**Purpose**: Thread-local context for API path overrides that don't pollute global environment.

**Fields**:
```python
@dataclass
class PathOverrideContext:
    root_dir: Path | None = None
    output_dir: Path | None = None
    data_dir: Path | None = None
    checkpoints_dir: Path | None = None
```

**Usage**:
```python
from dlkit.interfaces.api.overrides import path_override_context

# Apply path overrides in scope
with path_override_context({"output_dir": "./custom_output"}):
    result = train(settings)  # Uses custom_output

# Overrides cleared after context exit
```

**Implementation Notes**:
- Stored in thread-local storage (`threading.local()`)
- Context manager saves/restores previous context
- Overrides checked before falling back to environment/default
- Enables nested context managers with proper restoration

### Component 3: `resolve_with_context()`

**Purpose**: Resolve component paths with current override context.

**Parameters**:
- `component_path: str` - Path to resolve (e.g., "output/mlruns")
- `env: DLKitEnvironment | None = None` - Environment instance

**Returns**: `Path` - Resolved path respecting overrides

**Resolution Priority**:
1. Direct component override (e.g., `context.output_dir` for "output")
2. Prefix-based override (e.g., `context.output_dir` for "output/mlruns")
3. Environment root resolution
4. Default resolver system

**Example**:
```python
from dlkit.interfaces.api.overrides import resolve_with_context

# Without context
path = resolve_with_context("output/checkpoints")  # Uses environment

# With context
with path_override_context({"output_dir": "./my_output"}):
    path = resolve_with_context("output/checkpoints")
    # Returns: ./my_output/checkpoints
```

## Usage Patterns

### Basic Override Application
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")

# Apply overrides via API
result = train(
    settings,
    epochs=50,
    batch_size=64,
    output_dir="./experiment_1"
)
```

### Path Context Management
```python
from dlkit.interfaces.api.overrides import path_override_context

# Temporary path overrides
with path_override_context({
    "root_dir": "./experiment",
    "output_dir": "./experiment/output"
}):
    result = train(settings)
    # All paths resolved relative to overrides

# Context restored after exit
```

### Override Validation
```python
from dlkit.interfaces.api.overrides import basic_override_manager

# Validate before execution
errors = basic_override_manager.validate_overrides(
    settings,
    epochs=epochs,
    checkpoint_path=checkpoint_path
)

if errors:
    raise ValueError(f"Invalid overrides: {errors}")

# Safe to apply
new_settings = basic_override_manager.apply_overrides(settings, **overrides)
```

## Error Handling

**Validation Errors**:
```python
errors = basic_override_manager.validate_overrides(settings, **overrides)
# Returns list of error messages:
# - "checkpoint_path must be a valid path"
# - "Checkpoint file does not exist: ./missing.ckpt"
# - "epochs must be a positive number, got: -10"
# - "MLflow overrides require MLflow to be enabled in configuration"
```

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/api/test_overrides.py`
- Integration tests: `tests/integration/test_override_integration.py`

### Key Test Scenarios
1. **Override application**: Overrides correctly update settings
2. **Path context**: Thread-local context properly isolates overrides
3. **Validation**: Invalid overrides caught before application
4. **Immutability**: Original settings never mutated
5. **Nested contexts**: Context managers properly nest and restore

## Related Modules
- `dlkit.interfaces.api.commands`: Commands apply overrides before execution
- `dlkit.tools.config`: Settings models that are overridden
- `dlkit.tools.config.environment`: Environment configuration for path resolution

## Change Log
- **2025-10-03**: Added comprehensive documentation
- **2024-12-01**: Added path override context for API flexibility
- **2024-11-15**: Added override validation
- **2024-11-01**: Initial override manager implementation
