# CLI Commands Module

## Overview
The CLI commands module provides the primary command-line interface for DLKit operations through Typer-based commands. It implements training, inference, optimization, server management, configuration utilities, and checkpoint conversion workflows with rich console output and comprehensive parameter overrides.

## Architecture & Design Patterns
- **Command Pattern**: Each command module encapsulates a workflow (train, predict, optimize, etc.)
- **Adapter Pattern**: Delegates to API layer (`dlkit.interfaces.api`) for business logic
- **Separation of Concerns**: CLI handles presentation (Rich console, progress bars), API handles execution
- **Typer Framework**: Type-safe CLI with automatic help generation and validation
- **Error Handler Middleware**: Centralized error handling via `handle_api_error()`
- **Config Adapter**: Configuration loading abstraction for consistent path resolution

Key architectural decisions:
- CLI layer is thin - business logic lives in API layer
- Rich library for beautiful terminal output
- Direct parameter overrides for flexibility (--epochs, --batch-size, etc.)
- Consistent error handling across all commands
- Progress indicators for long-running operations

## Module Structure

### Public API (Command Groups)
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `train.app` | Typer | Training workflow commands | N/A |
| `predict.app` | Typer | Inference/prediction commands | N/A |
| `optimize.app` | Typer | Hyperparameter optimization commands | N/A |
| `server.app` | Typer | MLflow server management commands | N/A |
| `config.app` | Typer | Configuration utilities | N/A |
| `convert.app` | Typer | Checkpoint conversion to ONNX | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_run_training_impl` | Function | Core training execution logic | `None` |
| `_run_inference_impl` | Function | Core inference execution logic | `None` |
| `_run_optimization_impl` | Function | Core optimization execution logic | `None` |
| `_print_server_launch_summary` | Function | Server startup console output | `None` |
| `_handle_server_mode` | Function | Attached/detached server management | `None` |
| `_ensure_storage_setup_at_cli_level` | Function | MLflow storage initialization | `None` |
| `_display_server_status_table` | Function | Server status table rendering | `None` |
| `_display_config_table` | Function | Configuration table rendering | `None` |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api`: Core API functions (`train`, `predict`, `optimize`, `validate_config`)
- `dlkit.interfaces.cli.adapters.config_adapter`: Configuration loading (`load_config`)
- `dlkit.interfaces.cli.adapters.result_presenter`: Result display formatting
- `dlkit.interfaces.cli.middleware.error_handler`: Error handling (`handle_api_error`)
- `dlkit.interfaces.servers`: Server management services

### External Dependencies
- `typer`: CLI framework with type hints
- `rich`: Terminal formatting (console, progress, tables, panels, syntax highlighting)
- `pathlib`: Path handling

## Key Components

### Component 1: `train.py` - Training Commands

**Purpose**: Orchestrates model training workflows with configuration validation, parameter overrides, and MLflow tracking support.

**Main Command**: `dlkit train CONFIG.toml [options]`

**Parameters**:
- `config_path: Path` - Path to TOML configuration file
- `mlflow: bool = False` - Enable MLflow tracking
- `checkpoint: Path | None = None` - Resume training from checkpoint
- `validate_only: bool = False` - Validate config without training
- `root_dir: Path | None = None` - Root directory override
- `output_dir: Path | None = None` - Output directory override
- `data_dir: Path | None = None` - Data directory override
- `epochs: int | None = None` - Training epochs override
- `batch_size: int | None = None` - Batch size override
- `learning_rate: float | None = None` - Learning rate override
- `mlflow_host: str | None = None` - MLflow hostname override
- `mlflow_port: int | None = None` - MLflow port override
- `experiment_name: str | None = None` - Experiment name override
- `run_name: str | None = None` - Run name override

**Returns**: `None` (exits with status code)

**Raises**:
- `typer.Exit`: On validation failures or errors
- `DLKitError`: Training/configuration errors (handled by middleware)

**Example**:
```bash
# Basic training
dlkit train config.toml

# With MLflow tracking
dlkit train config.toml --mlflow --experiment-name my_exp

# Resume from checkpoint with parameter overrides
dlkit train config.toml --checkpoint model.ckpt --epochs 100 --batch-size 64

# Validate configuration only
dlkit train config.toml --validate-only
```

**Implementation Notes**:
- Loads config via `load_config()` adapter
- Validates configuration using `validate_config()` API
- Progress bars with Rich `Progress` for visual feedback
- All overrides passed through to `api_train()` function
- Graceful error handling with user-friendly messages
- Presents results via `present_training_result()`

---

### Component 2: `predict.py` - Prediction/Inference Commands

**Purpose**: Runs Lightning-based predictions using trained checkpoints and training configurations.

**Main Command**: `dlkit predict CONFIG.toml CHECKPOINT [options]`

**Parameters**:
- `config_path: Path` - Path to TOML configuration file
- `checkpoint: Path` - Path to model checkpoint (.ckpt)
- `root_dir: Path | None = None` - Root directory override
- `output_dir: Path | None = None` - Output directory override
- `data_dir: Path | None = None` - Data directory override
- `batch_size: int | None = None` - Batch size override
- `save_predictions: bool = True` - Save predictions to file

**Returns**: `None` (exits with status code)

**Raises**:
- `typer.Exit`: If checkpoint not found or errors occur
- `DLKitError`: Inference errors (handled by middleware)

**Example**:
```bash
# Basic prediction
dlkit predict config.toml model.ckpt

# With overrides
dlkit predict config.toml model.ckpt --output-dir ./results --batch-size 128

# Without saving predictions
dlkit predict config.toml model.ckpt --no-save
```

**Implementation Notes**:
- Checkpoint path required as CLI argument
- Can fallback to `[MODEL].checkpoint` from config
- Uses training config for data loading consistency
- Progress indicators during inference
- Results presented via `present_inference_result()`

---

### Component 3: `optimize.py` - Hyperparameter Optimization Commands

**Purpose**: Executes Optuna-based hyperparameter optimization with optional MLflow tracking.

**Main Command**: `dlkit optimize CONFIG.toml --trials N [options]`

**Parameters**:
- `config_path: Path` - Path to TOML configuration file
- `trials: int = 100` - Number of optimization trials
- `study_name: str | None = None` - Optuna study name
- `mlflow: bool = False` - Enable MLflow tracking
- `root_dir: Path | None = None` - Root directory override
- `output_dir: Path | None = None` - Output directory override

**Subcommands**:
- `dlkit optimize status STUDY_NAME STORAGE` - Show study status
- `dlkit optimize plot STUDY_NAME STORAGE` - Generate optimization plots

**Returns**: `None` (exits with status code)

**Raises**:
- `typer.Exit`: If Optuna not enabled or errors occur
- `DLKitError`: Optimization errors (handled by middleware)

**Example**:
```bash
# Run optimization
dlkit optimize config.toml --trials 50 --study-name my_study

# With MLflow tracking
dlkit optimize config.toml --trials 100 --mlflow

# Check study status
dlkit optimize status my_study sqlite:///study.db

# Generate plots
dlkit optimize plot my_study sqlite:///study.db --type param_importances
```

**Implementation Notes**:
- Requires `[OPTUNA].enabled = true` in config
- Validates Optuna configuration before execution
- Supports study resumption via same `--study-name`
- Integrates with MLflow for trial tracking
- Rich tables for study status display
- Plot types: optimization_history, param_importances, parallel_coordinate, slice

---

### Component 4: `server.py` - MLflow Server Management

**Purpose**: Manages MLflow tracking server lifecycle (start, stop, status, info) with interactive storage setup.

**Subcommands**:
- `dlkit server start [CONFIG] [options]` - Start MLflow server
- `dlkit server stop [options]` - Stop MLflow server
- `dlkit server status [options]` - Check server status
- `dlkit server info [CONFIG]` - Show server configuration

**Start Parameters**:
- `config_path: Path | None = None` - Optional configuration file
- `host: str | None = None` - Server hostname override
- `port: int | None = None` - Server port override
- `backend_store_uri: str | None = None` - Backend store override
- `artifacts_destination: str | None = None` - Artifacts path override
- `detach: bool = False` - Run server in background

**Returns**: `None` (exits with status code)

**Raises**:
- `typer.Exit`: On startup/shutdown failures

**Example**:
```bash
# Start with defaults
dlkit server start

# Start with config and overrides
dlkit server start config.toml --host 0.0.0.0 --port 8080 --detach

# Check server status
dlkit server status --port 5000

# Stop server
dlkit server stop --host localhost --port 5000

# Show server info from config
dlkit server info config.toml
```

**Implementation Notes**:
- Interactive storage setup via `_ensure_storage_setup_at_cli_level()`
- User confirmation for creating default `mlruns/` directory
- Attached mode with Ctrl+C graceful shutdown
- Detached mode for background execution
- Health checks and connectivity verification
- Rich panels for server information display
- Delegates to `ServerApplicationService` for business logic

---

### Component 5: `config.py` - Configuration Utilities

**Purpose**: Provides configuration validation, inspection, template creation, and synchronization commands.

**Subcommands**:
- `dlkit config validate CONFIG` - Validate configuration
- `dlkit config show CONFIG` - Display configuration
- `dlkit config create --output PATH --type TYPE` - Create template
- `dlkit config sync-templates` - Sync example templates

**Validate Parameters**:
- `config_path: Path` - Configuration file to validate
- `strategy: str | None = None` - Strategy type (training, mlflow, optuna, inference)

**Show Parameters**:
- `config_path: Path` - Configuration file to display
- `section: str | None = None` - Specific section to show
- `format: str = "table"` - Output format (json, yaml, table)

**Create Parameters**:
- `output_path: Path` - Output path for template
- `template_type: str = "training"` - Template type

**Returns**: `None` (exits with status code)

**Example**:
```bash
# Validate configuration
dlkit config validate config.toml --strategy mlflow

# Show full config as table
dlkit config show config.toml

# Show specific section as JSON
dlkit config show config.toml --section MODEL --format json

# Create training template
dlkit config create --output new_config.toml --type training

# Sync templates
dlkit config sync-templates --write
```

**Implementation Notes**:
- Auto-detects strategy if not specified (inference vs training)
- Multiple output formats with Rich syntax highlighting
- Hierarchical table display for nested configurations
- Template generation via `generate_template()` API
- Template sync for maintaining consistency across examples
- Drift detection with `--check` flag

---

### Component 6: `convert.py` - Checkpoint Conversion

**Purpose**: Converts Lightning checkpoints to ONNX format for deployment.

**Main Command**: `dlkit convert CHECKPOINT OUTPUT [options]`

**Parameters**:
- `checkpoint: Path` - Path to model checkpoint (.ckpt)
- `output: Path` - Output ONNX file path (.onnx)
- `config: Path | None = None` - Optional config for shape inference
- `shape: str | None = None` - Feature dimensions (e.g., "3,224,224")
- `batch_size: int | None = 1` - Batch size for export
- `opset: int = 17` - ONNX opset version

**Returns**: `None` (exits with status code)

**Raises**:
- `typer.Exit`: If neither shape nor config provided, or conversion fails
- `typer.BadParameter`: If required parameters missing

**Example**:
```bash
# Convert with explicit shape
dlkit convert model.ckpt model.onnx --shape 3,224,224

# Convert using config for shape inference
dlkit convert model.ckpt model.onnx --config config.toml

# Specify batch size and opset
dlkit convert model.ckpt model.onnx --shape 3,32,32 --batch-size 4 --opset 17
```

**Implementation Notes**:
- Requires either `--shape` or `--config` for input dimensions
- Delegates to `ConvertCommand` for actual conversion
- Rich panel output showing export details
- POSIX path formatting for cross-platform consistency

## Usage Patterns

### Common Use Case 1: Training Workflow
```bash
# 1. Validate configuration
dlkit config validate config.toml

# 2. Train model
dlkit train config.toml --mlflow --epochs 50

# 3. Resume if needed
dlkit train config.toml --checkpoint outputs/model.ckpt --epochs 100
```

### Common Use Case 2: Optimization Workflow
```bash
# 1. Configure Optuna in config.toml
[OPTUNA]
enabled = true
n_trials = 100

# 2. Run optimization
dlkit optimize config.toml --trials 50 --mlflow --study-name exp_v1

# 3. Check progress
dlkit optimize status exp_v1 sqlite:///study.db

# 4. Visualize results
dlkit optimize plot exp_v1 sqlite:///study.db --type param_importances
```

### Common Use Case 3: MLflow Server Management
```bash
# 1. Start server
dlkit server start --host 0.0.0.0 --port 5000 --detach

# 2. Check status
dlkit server status --port 5000

# 3. Train with tracking
dlkit train config.toml --mlflow --mlflow-port 5000

# 4. Stop server
dlkit server stop --port 5000
```

### Common Use Case 4: Inference and Export
```bash
# 1. Run predictions
dlkit predict config.toml model.ckpt --save

# 2. Convert to ONNX
dlkit convert model.ckpt model.onnx --config config.toml

# 3. Verify ONNX model externally
```

## Error Handling

**Exceptions Raised**:
- `typer.Exit`: Explicit CLI exit with status codes (0=success, 1=failure)
- `DLKitError`: Domain errors from API layer (handled by middleware)
- `typer.BadParameter`: Invalid parameter values
- Generic `Exception`: Unexpected errors

**Error Handling Pattern**:
```python
try:
    # Command execution
    result = api_function(settings, **overrides)
    present_result(result, console)
except typer.Exit:
    raise  # Propagate explicit exits
except DLKitError as e:
    handle_api_error(e, console)  # User-friendly error messages
    raise typer.Exit(1)
except Exception as e:
    console.print(f"[red]Unexpected error: {e}[/red]")
    raise typer.Exit(1)
```

**User-Friendly Error Display**:
- Rich console formatting for error messages
- Contextual help suggestions (e.g., "Enable [OPTUNA] in config")
- Validation errors with specific field information
- Server errors with troubleshooting steps

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/cli/test_commands.py` (hypothetical)
- Integration tests: CLI invocation with test configs
- Error handling: Invalid configs, missing checkpoints, server failures

### Key Test Scenarios
1. **Training command**: Valid config, checkpoint resumption, validation-only mode
2. **Prediction command**: Checkpoint loading, parameter overrides
3. **Optimization command**: Study creation, status checking, plotting
4. **Server commands**: Start/stop lifecycle, status checks, config display
5. **Config commands**: Validation, template creation, sync operations
6. **Convert command**: ONNX export with shape/config inference
7. **Error handling**: Invalid configs, missing files, API errors

### Fixtures Used
- `tmp_path`: Temporary paths for outputs
- `sample_config`: Example configuration files
- `mock_api_functions`: Mocked API calls for isolated testing
- `cli_runner`: Typer CliRunner for command invocation

## Performance Considerations
- Lazy imports for optional dependencies (Optuna, plotly)
- Progress bars only for long operations (training, optimization)
- Efficient config loading with caching
- Rich console rendering optimized for terminal output
- Background server mode for non-blocking execution

## Future Improvements / TODOs
- [ ] Add `dlkit resume` command for automatic checkpoint detection
- [ ] Batch prediction mode for multiple checkpoints
- [ ] Configuration diffing: `dlkit config diff config1.toml config2.toml`
- [ ] Interactive configuration wizard: `dlkit config wizard`
- [ ] Export command support for TorchScript, CoreML
- [ ] Server command: `dlkit server logs` for viewing server logs
- [ ] Optimization: Real-time trial visualization with `--watch` flag
- [ ] Shell completion generation: `dlkit completion bash/zsh/fish`

## Related Modules
- `dlkit.interfaces.api`: API layer that CLI commands delegate to
- `dlkit.interfaces.cli.adapters`: Configuration and result presentation adapters
- `dlkit.interfaces.cli.middleware`: Error handling middleware
- `dlkit.interfaces.servers`: Server management services
- `dlkit.tools.config`: Configuration models and validation

## Change Log
- **2025-10-03**: Comprehensive CLI commands documentation created
- **2024-10-02**: Added convert command for ONNX export
- **2024-09-30**: Refactored server commands to use application service
- **2024-09-24**: Initial CLI structure with train, predict, optimize commands
