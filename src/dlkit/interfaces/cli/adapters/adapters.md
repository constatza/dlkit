# CLI Adapters Module

## Overview
The CLI adapters module provides abstraction layers between the command-line interface and the core API. It handles configuration loading with proper error handling and presents workflow results in beautiful, user-friendly terminal output using the Rich library. This module follows the Adapter pattern to decouple CLI presentation concerns from business logic.

## Architecture & Design Patterns
- **Adapter Pattern**: Translates between CLI needs and API/config system interfaces
- **Dependency Inversion Principle**: Depends on protocol abstractions (`BaseSettingsProtocol`)
- **Separation of Concerns**: Configuration loading separate from result presentation
- **Error Translation**: Converts low-level errors to user-friendly `ConfigurationError`
- **Context Manager Pattern**: Uses `path_override_context` for temporary path overrides
- **Presentation Layer**: Rich library for terminal output formatting

Key architectural decisions:
- Adapters are pure functions (no state)
- Configuration loading validates early and fails fast
- Result presentation is read-only (no side effects on results)
- Path overrides via context managers (no settings mutation)
- Protocol-based design enables testing with mocks

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `load_config` | Function | Load configuration with CLI error handling | `BaseSettingsProtocol` |
| `validate_config_path` | Function | Validate config file path and accessibility | `bool` |
| `present_training_result` | Function | Display training results with Rich formatting | `None` |
| `present_inference_result` | Function | Display inference results with Rich formatting | `None` |
| `present_optimization_result` | Function | Display optimization results with Rich formatting | `None` |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_display_metrics_table` | Function | Render metrics as Rich table | `None` |
| `_display_artifacts_table` | Function | Render artifacts as Rich table | `None` |
| `_display_best_parameters` | Function | Render best parameters as Rich table | `None` |
| `_display_study_summary` | Function | Render study summary as Rich table | `None` |
| `_display_prediction_summary` | Function | Render prediction summary with details | `None` |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.domain`: Domain models (`TrainingResult`, `InferenceResult`, `OptimizationResult`)
- `dlkit.interfaces.api.domain.errors`: Error types (`ConfigurationError`)
- `dlkit.tools.config`: Configuration loading (`load_settings`, `load_sections`)
- `dlkit.tools.config.protocols`: Protocol interfaces (`BaseSettingsProtocol`)
- `dlkit.interfaces.api.overrides.path_context`: Path override context manager
- `dlkit.core.postprocessing`: Prediction summarization utilities

### External Dependencies
- `rich`: Terminal formatting (Console, Panel, Table, Text, Syntax)
- `pathlib`: Path handling
- `typing`: Type annotations

## Key Components

### Component 1: `load_config`

**Purpose**: Load configuration from TOML file with CLI-specific error handling, partial loading optimization, and path override support.

**Parameters**:
- `config_path: Path` - Path to configuration file (must exist)
- `root_dir: Path | None = None` - Optional root directory override
- `output_dir: Path | None = None` - Optional output directory override (deprecated)
- `workflow_type: str | None = None` - Workflow type for partial loading ('training', 'inference', or None)

**Returns**: `BaseSettingsProtocol` - Loaded settings object appropriate for workflow type

**Raises**:
- `ConfigurationError`: If file not found, invalid format, or loading fails

**Example**:
```python
from pathlib import Path
from dlkit.interfaces.cli.adapters.config_adapter import load_config

# Basic loading
settings = load_config(Path("config.toml"))

# With workflow-specific partial loading
training_settings = load_config(Path("config.toml"), workflow_type="training")

# With root directory override
settings = load_config(Path("config.toml"), root_dir=Path("/custom/root"), workflow_type="training")
```

**Implementation Notes**:
- Validates file existence before attempting load
- Uses partial loading for performance (only loads required sections)
- Applies root_dir via `path_override_context` (temporary, no mutation)
- Translates all exceptions to `ConfigurationError` with context
- Supports training/inference workflow types or defaults to training
- Thread-safe path override via context manager

---

### Component 2: `validate_config_path`

**Purpose**: Validate configuration file path, accessibility, and format before loading.

**Parameters**:
- `config_path: Path` - Path to configuration file

**Returns**: `bool` - Always returns `True` (raises on validation failure)

**Raises**:
- `ConfigurationError`: If path invalid, not a file, unsupported format, or inaccessible

**Example**:
```python
from pathlib import Path
from dlkit.interfaces.cli.adapters.config_adapter import validate_config_path

try:
    validate_config_path(Path("config.toml"))
    print("Config path valid!")
except ConfigurationError as e:
    print(f"Validation failed: {e}")
```

**Implementation Notes**:
- Checks file existence
- Validates is a file (not directory)
- Verifies supported extension (.toml, .json, .yaml, .yml)
- Tests read permissions by reading 1 character
- Provides detailed error context for debugging
- Used for early validation before expensive operations

---

### Component 3: `present_training_result`

**Purpose**: Display training workflow results in formatted, user-friendly terminal output with Rich library.

**Parameters**:
- `result: TrainingResult` - Training result from API
- `console: Console` - Rich console for output

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit.interfaces.api import train
from dlkit.interfaces.cli.adapters.result_presenter import present_training_result

console = Console()
result = train(settings)
present_training_result(result, console)
```

**Output Format**:
```
╭─ Training Summary ─────────────────╮
│ 🏋️ Training Results                │
│                                    │
│ Duration: 125.34 seconds           │
│ Metrics: 5 recorded                │
│ Artifacts: 3 saved                 │
╰────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ train_loss      │ 0.123456  │
│ val_loss        │ 0.234567  │
│ best_val_loss   │ 0.220000  │
└─────────────────┴───────────┘

┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Artifact   ┃ Path                 ┃ Exists ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ checkpoint │ outputs/model.ckpt   │ ✅     │
│ config     │ outputs/config.toml  │ ✅     │
└────────────┴──────────────────────┴────────┘
```

**Implementation Notes**:
- Creates summary panel with duration, metric count, artifact count
- Displays metrics table with 6 decimal precision for floats
- Shows artifacts table with existence verification
- Color-coded for visual clarity (green=success, cyan=info, yellow=warning)
- Non-destructive: doesn't modify result object

---

### Component 4: `present_inference_result`

**Purpose**: Display inference workflow results including predictions, metrics, and summary statistics.

**Parameters**:
- `result: InferenceResult` - Inference result from API
- `console: Console` - Rich console for output
- `save_predictions: bool = True` - Whether predictions were saved to disk

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit import load_model
from dlkit.interfaces.cli.adapters.result_presenter import present_inference_result

console = Console()
with load_model("model.ckpt", device="auto") as predictor:
    result = predictor.predict(x=input_tensor)
present_inference_result(result, console, save_predictions=True)
```

**Output Format**:
```
╭─ Inference Summary ────────────────╮
│ 🔮 Inference Results               │
│                                    │
│ Duration: 15.67 seconds            │
│ Predictions: Generated successfully│
│ Predictions saved to output dir    │
│ Metrics: 2 recorded                │
╰────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ inference_time  │ 15.670000 │
│ batch_size      │ 64        │
└─────────────────┴───────────┘

╭─ Predictions ──────────────────────╮
│ 🔮 Prediction Summary              │
│ Generated 1000 predictions         │
╰────────────────────────────────────╯
```

**Implementation Notes**:
- Shows prediction generation status
- Conditionally displays save confirmation
- Displays inference metrics if available
- Prediction summary with length/type info
- Optional verbose mode via `DLKIT_CLI_VERBOSE` env var
- Verbose mode shows shape, dtype, graph statistics using `summarize()` utility

---

### Component 5: `present_optimization_result`

**Purpose**: Display hyperparameter optimization results including best trial, parameters, and study statistics.

**Parameters**:
- `result: OptimizationResult` - Optimization result from API
- `console: Console` - Rich console for output

**Returns**: `None` (side effect: prints to console)

**Example**:
```python
from rich.console import Console
from dlkit.interfaces.api import optimize
from dlkit.interfaces.cli.adapters.result_presenter import present_optimization_result

console = Console()
result = optimize(settings, trials=100)
present_optimization_result(result, console)
```

**Output Format**:
```
╭─ Optimization Summary ─────────────╮
│ ⚡ Optimization Results            │
│                                    │
│ Duration: 3600.45 seconds          │
│ Best value: 0.123456               │
│ Best trial: #42                    │
╰────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Parameter      ┃ Value    ┃ Type   ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ learning_rate  │ 0.001    │ float  │
│ batch_size     │ 64       │ int    │
│ dropout        │ 0.3      │ float  │
└────────────────┴──────────┴────────┘

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Statistic       ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ total_trials    │ 100     │
│ complete_trials │ 95      │
│ failed_trials   │ 2       │
│ pruned_trials   │ 3       │
└─────────────────┴─────────┘

┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ best_val_loss   │ 0.123456  │
└─────────────────┴───────────┘
```

**Implementation Notes**:
- Displays best trial value and number
- Shows best parameters with type information
- Includes study summary statistics
- Shows final training metrics from best trial
- Color-coded panels (magenta for optimization theme)
- Gracefully handles missing fields (shows "N/A")

---

### Component 6: Internal Display Functions

**Purpose**: Helper functions for rendering specific data structures as Rich tables.

**Functions**:
- `_display_metrics_table(metrics, console, title)` - Metrics with 6-digit float formatting
- `_display_artifacts_table(artifacts, console)` - Artifacts with existence checks
- `_display_best_parameters(params, console)` - Parameters with type information
- `_display_study_summary(summary, console)` - Study statistics
- `_display_prediction_summary(predictions, console)` - Prediction details with optional verbose mode

**Implementation Notes**:
- Consistent table styling across all displays
- Float formatting: 6 decimal places for metrics
- Path verification for artifacts (✅/❌ indicators)
- Type introspection for parameters
- Prediction summary supports tensors, lists, dicts, graphs via `summarize()` utility
- Verbose mode controlled by `DLKIT_CLI_VERBOSE` environment variable

## Usage Patterns

### Common Use Case 1: Configuration Loading in CLI Commands
```python
from pathlib import Path
from dlkit.interfaces.cli.adapters.config_adapter import load_config
from dlkit.interfaces.api.domain.errors import ConfigurationError

try:
    # Load with workflow-specific optimization
    settings = load_config(
        Path("config.toml"), root_dir=Path("/custom/root"), workflow_type="training"
    )
except ConfigurationError as e:
    console.print(f"[red]Config error: {e.message}[/red]")
    raise typer.Exit(1)
```

### Common Use Case 2: Presenting Training Results
```python
from rich.console import Console
from dlkit.interfaces.api import train
from dlkit.interfaces.cli.adapters.result_presenter import present_training_result

console = Console()

# Execute training
result = train(settings, mlflow=True)

# Present results with rich formatting
console.print("🎉 Training completed successfully!")
present_training_result(result, console)
```

### Common Use Case 3: Early Configuration Validation
```python
from pathlib import Path
from dlkit.interfaces.cli.adapters.config_adapter import validate_config_path
from dlkit.interfaces.api.domain.errors import ConfigurationError

try:
    # Validate before expensive operations
    validate_config_path(Path("config.toml"))
    console.print("✅ Config path validated")
except ConfigurationError as e:
    console.print(f"[red]Invalid config path: {e.message}[/red]")
    raise typer.Exit(1)
```

### Common Use Case 4: Verbose Prediction Display
```bash
# Set verbose mode for detailed prediction info
export DLKIT_CLI_VERBOSE=1
dlkit predict config.toml model.ckpt
```

```python
# Programmatic verbose display
import os

os.environ["DLKIT_CLI_VERBOSE"] = "1"

present_inference_result(result, console, save_predictions=True)
# Now shows shape, dtype, graph statistics, etc.
```

## Error Handling

**Exceptions Raised**:
- `ConfigurationError`: All configuration loading/validation errors
  - File not found
  - Invalid format (not .toml, .json, .yaml, .yml)
  - Permission denied
  - Parse errors
  - Invalid configuration structure

**Error Context**: All `ConfigurationError` exceptions include context dict:
```python
ConfigurationError(
    message="File not found: config.toml", context={"config_path": "/path/to/config.toml"}
)
```

**Error Handling Pattern**:
```python
try:
    settings = load_config(config_path, workflow_type="training")
except ConfigurationError as e:
    # Access structured error information
    console.print(f"[red]Error: {e.message}[/red]")
    if e.context:
        console.print(f"Context: {e.context}")
    raise typer.Exit(1)
```

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/cli/adapters/test_config_adapter.py`
- Unit tests: `tests/interfaces/cli/adapters/test_result_presenter.py`
- Integration tests: End-to-end CLI testing with real configs

### Key Test Scenarios
1. **Config loading**: Valid TOML, workflow-specific loading, path overrides
2. **Config validation**: Missing files, invalid formats, permission errors
3. **Result presentation**: Training/inference/optimization results with various data
4. **Error handling**: Configuration errors with proper context
5. **Verbose mode**: Prediction summary with DLKIT_CLI_VERBOSE enabled
6. **Edge cases**: Empty metrics, missing artifacts, null predictions

### Fixtures Used
- `tmp_path`: Temporary paths for test configs
- `sample_training_result`: Mock training results
- `sample_inference_result`: Mock inference results
- `sample_optimization_result`: Mock optimization results
- `mock_console`: Captured Rich console output

## Performance Considerations
- Partial configuration loading for workflow-specific sections
- Path override via context manager (no deep copying)
- Lazy result formatting (only formats visible fields)
- Efficient Rich table rendering with streaming
- No redundant file I/O during validation
- Verbose mode opt-in to avoid overhead

## Future Improvements / TODOs
- [ ] Support for YAML/JSON configuration files (currently TOML-focused)
- [ ] Interactive configuration editing via Rich prompts
- [ ] Progress bars for large prediction summaries
- [ ] Export results to JSON/CSV via `--export` flag
- [ ] Configurable float precision for metrics display
- [ ] Artifact download links for MLflow-tracked runs
- [ ] Comparison view: `present_comparison_result(results1, results2)`
- [ ] Plot generation: inline terminal plots with rich-pixels/plotille

## Related Modules
- `dlkit.interfaces.api.domain`: Result models and error types
- `dlkit.interfaces.cli.commands`: CLI commands that use these adapters
- `dlkit.interfaces.cli.middleware`: Error handler that consumes ConfigurationError
- `dlkit.tools.config`: Configuration loading infrastructure
- `dlkit.core.postprocessing`: Prediction summarization utilities

## Change Log
- **2025-10-03**: Comprehensive CLI adapters documentation created
- **2024-10-02**: Added verbose mode for prediction summaries
- **2024-09-30**: Migrated to protocol-based config loading
- **2024-09-24**: Initial adapters with config loading and result presentation
