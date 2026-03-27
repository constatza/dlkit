# Commands Module

## Overview
The commands module implements the Command Pattern for DLKit's API layer, providing structured execution of workflow operations with validation, error handling, and logging. Each command encapsulates a specific workflow (training, inference, optimization, configuration) as an executable unit with clear input/output contracts.

## Architecture & Design Patterns
- **Command Pattern**: Encapsulates requests as objects with `execute()` method for uniform invocation
- **Dependency Injection**: Commands receive dependencies (services, managers) via constructor
- **Strategy Pattern**: Input validation and execution strategies vary by command type
- **Registry Pattern**: `CommandDispatcher` maintains command registry for dynamic lookup
- **Immutable Input**: `@dataclass(frozen=True)` input objects prevent mutation during execution
- **Direct Exception Raising**: Commands raise domain exceptions directly rather than returning error wrappers

Key architectural decisions:
- Commands are stateless - all state passed via input data
- Each command has a single responsibility (SRP)
- Commands depend on service abstractions, not concrete implementations (DIP)
- Input validation separated from execution logic for clarity

## Module Structure

### Public API
| Name | File | Type | Purpose | Returns |
|------|------|------|---------|---------|
| `BaseCommand[TInput, TOutput]` | `base.py` | Abstract Class | Base command with validation and execution contracts | N/A |
| `TrainCommand` | `train_command.py` | Class | Execute training workflows | `TrainingResult` |
| `InferenceCommand` | `inference_command.py` | Class | Execute inference workflows | `InferenceResult` |
| `OptimizationCommand` | `optimization_command.py` | Class | Execute Optuna optimization workflows | `OptimizationResult` |
| `ValidationCommand` | `validation_command.py` | Class | Validate configuration against strategies | `bool` |
| `ConvertCommand` | `convert_command.py` | Class | Convert checkpoints to ONNX format | `ConvertResult` |
| `GenerateTemplateCommand` | `configuration_command.py` | Class | Generate configuration templates | `GenerateTemplateCommandOutput` |
| `ValidateTemplateCommand` | `configuration_command.py` | Class | Validate configuration templates | `ValidateTemplateCommandOutput` |
| `CommandDispatcher` | `dispatcher.py` | Class | Route and execute commands by name | `Any` |
| `get_dispatcher()` | `dispatcher.py` | Function | Get global dispatcher instance | `CommandDispatcher` |
| `OverrideNormalizer` | `normalizer.py` | Class | Pure utility for path normalization and None-filtering | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `CommandRegistry` | Dataclass | Registry for command implementations | N/A |
| `TrainCommandInput` | Dataclass | Input for training command | N/A |
| `InferenceCommandInput` | Dataclass | Input for inference command | N/A |
| `OptimizationCommandInput` | Dataclass | Input for optimization command | N/A |
| `ValidationCommandInput` | Dataclass | Input for validation command | N/A |
| `ConvertCommandInput` | Dataclass | Input for model conversion command | N/A |
| `_build_overrides_dict()` | Method | Build overrides from input data | `dict[str, Any]` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `BaseCommand[TInput, TOutput]` | `execute()`, `validate_input()` | Abstract command interface |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.services`: Business logic services (TrainingService, InferenceService, OptimizationService, ConfigurationService)
- `dlkit.interfaces.api.services.override_service`: `basic_override_manager` singleton
- `dlkit.interfaces.api.commands.normalizer`: `OverrideNormalizer` — path normalization and None-filtering
- `dlkit.domain`: Shared result types (TrainingResult, InferenceResult, OptimizationResult)
- `dlkit.interfaces.api.domain`: Error types (WorkflowError, ConfigurationError, etc.)
- `dlkit.tools.config`: Configuration models (GeneralSettings, BaseSettingsProtocol)
- `dlkit.tools.utils`: Logging and error handling utilities
- `dlkit.runtime.workflows`: Component factories and build strategies

### External Dependencies
- `loguru`: Structured logging
- `torch`: PyTorch for model conversion
- `pydantic`: Data validation for command inputs
- `tomlkit`: TOML parsing for template validation

## Key Components

### Component 1: `BaseCommand[TInput, TOutput]`

**Purpose**: Abstract base class defining the contract for all commands. Enforces separation of validation and execution logic while providing consistent interface.

**Type Parameters**:
- `TInput` - Type of command input data
- `TOutput` - Type of command execution result

**Methods**:
- `__init__(command_name: str) -> None` - Initialize with human-readable identifier
- `execute(input_data: TInput, settings: BaseSettingsProtocol, **kwargs: Any) -> TOutput` - Execute the command (abstract)
- `validate_input(input_data: TInput, settings: BaseSettingsProtocol) -> None` - Validate input before execution (abstract)

**Example**:
```python
class MyCommand(BaseCommand[MyInput, MyOutput]):
    def __init__(self) -> None:
        super().__init__("my_command")

    def validate_input(self, input_data: MyInput, settings: BaseSettingsProtocol) -> None:
        if not input_data.required_field:
            raise WorkflowError("required_field is missing")

    def execute(
        self, input_data: MyInput, settings: BaseSettingsProtocol, **kwargs: Any
    ) -> MyOutput:
        self.validate_input(input_data, settings)
        # Execute command logic
        return MyOutput(result="success")
```

**Implementation Notes**:
- Generic type parameters enable type-safe command implementations
- Validation is called explicitly at start of `execute()` for fail-fast behavior
- Commands are stateless - all data flows through parameters

---

### Component 2: `TrainCommand`

**Purpose**: Command for executing training workflows with parameter overrides, checkpoint resumption, and MLflow tracking support.

**Constructor Parameters**:
- `command_name: str = "train"` - Command identifier for dispatcher

**Key Attributes**:
- `override_manager: BasicOverrideManager` - Manages parameter overrides
- `training_service: TrainingService` - Executes training logic

**Key Methods**:
- `validate_input(input_data: TrainCommandInput, settings: BaseSettingsProtocol) -> None` - Validate training parameters and overrides
- `execute(input_data: TrainCommandInput, settings: BaseSettingsProtocol, **kwargs: Any) -> TrainingResult` - Execute training workflow
- `_build_overrides_dict(input_data: TrainCommandInput) -> dict[str, Any]` - Convert input to overrides dictionary

**Input Fields** (`TrainCommandInput`):
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
- `additional_overrides: dict[str, Any] = field(default_factory=dict)` - Extra overrides

**Returns**: `TrainingResult` - Contains model state, metrics, artifacts, duration

**Raises**:
- `WorkflowError` - On validation failure or execution error

**Example**:
```python
from dlkit.interfaces.api.commands import TrainCommand, TrainCommandInput
from dlkit.tools.config import GeneralSettings

# Create command
command = TrainCommand()

# Prepare input
input_data = TrainCommandInput(
    mlflow=True, epochs=50, batch_size=32, learning_rate=0.001, experiment_name="my_experiment"
)

# Execute
settings = GeneralSettings.from_toml("config.toml")
result = command.execute(input_data, settings)

print(f"Training completed in {result.duration_seconds:.2f}s")
print(f"Best checkpoint: {result.checkpoint_path}")
```

**Implementation Notes**:
- Validates overrides before applying them using `override_manager`
- Applies overrides to settings via `BasicSettings.patch()` for immutability
- Checkpoint path extracted from overrides for service call
- Comprehensive logging at info level for execution tracking
- Error context includes command name for debugging

---

### Component 3: `InferenceCommand`

**Purpose**: Command for executing inference workflows with checkpoint loading and parameter overrides.

**Constructor Parameters**:
- `command_name: str = "infer"` - Command identifier

**Key Attributes**:
- `override_manager: BasicOverrideManager` - Manages parameter overrides
- `inference_service: InferenceService` - Executes inference logic

**Key Methods**:
- `validate_input(input_data: InferenceCommandInput, settings: BaseSettingsProtocol) -> None` - Validate checkpoint path and overrides
- `execute(input_data: InferenceCommandInput, settings: BaseSettingsProtocol, **kwargs: Any) -> InferenceResult` - Execute inference workflow
- `_build_overrides_dict(input_data: InferenceCommandInput) -> dict[str, Any]` - Convert input to overrides dictionary

**Input Fields** (`InferenceCommandInput`):
- `checkpoint_path: Path | str` - Required checkpoint path
- `root_dir: Path | str | None = None` - Override root directory
- `output_dir: Path | str | None = None` - Override output directory
- `data_dir: Path | str | None = None` - Override data directory
- `batch_size: int | None = None` - Override batch size
- `additional_overrides: dict[str, Any] | None = None` - Extra overrides

**Returns**: `InferenceResult` - Contains predictions, metrics, duration

**Raises**:
- `WorkflowError` - On missing checkpoint or execution error

**Example**:
```python
from dlkit.interfaces.api.commands import InferenceCommand, InferenceCommandInput

command = InferenceCommand()

input_data = InferenceCommandInput(
    checkpoint_path="./checkpoints/best.ckpt", batch_size=64, output_dir="./inference_output"
)

result = command.execute(input_data, settings)
predictions = result.predictions
```

**Implementation Notes**:
- Checkpoint path is required - validation fails if missing
- `__post_init__` ensures `additional_overrides` is never None
- String paths automatically converted to `Path` objects
- Overrides validated before application

---

### Component 4: `OptimizationCommand`

**Purpose**: Command for executing Optuna hyperparameter optimization workflows with trial configuration and tracking.

**Constructor Parameters**:
- `command_name: str = "optimize"` - Command identifier

**Key Attributes**:
- `override_manager: BasicOverrideManager` - Manages parameter overrides
- `optimization_service: OptimizationService` - Executes optimization logic

**Key Methods**:
- `validate_input(input_data: OptimizationCommandInput, settings: BaseSettingsProtocol) -> None` - Validate trials count and overrides
- `execute(input_data: OptimizationCommandInput, settings: BaseSettingsProtocol, **kwargs: Any) -> OptimizationResult` - Execute optimization workflow
- `_build_overrides_dict(input_data: OptimizationCommandInput) -> dict[str, Any]` - Convert input to overrides dictionary

**Input Fields** (`OptimizationCommandInput`):
- `trials: int | None = None` - Number of optimization trials
- `mlflow: bool = False` - Enable MLflow tracking
- `checkpoint_path: Path | str | None = None` - Warm-start checkpoint
- `root_dir: Path | str | None = None` - Override root directory
- `output_dir: Path | str | None = None` - Override output directory
- `data_dir: Path | str | None = None` - Override data directory
- `study_name: str | None = None` - Override Optuna study name
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `additional_overrides: dict[str, Any] | None = None` - Extra overrides

**Returns**: `OptimizationResult` - Contains best trial, training result, study summary, duration

**Raises**:
- `WorkflowError` - On invalid trials count or execution error

**Example**:
```python
from dlkit.interfaces.api.commands import OptimizationCommand, OptimizationCommandInput

command = OptimizationCommand()

input_data = OptimizationCommandInput(
    trials=100, mlflow=True, study_name="hyperparameter_search", experiment_name="optimization_exp"
)

result = command.execute(input_data, settings)
print(f"Best trial: {result.best_trial}")
print(f"Best parameters: {result.study_summary}")
```

**Implementation Notes**:
- Validates trials count is positive before execution
- Only overrides trials if different from default (100)
- Checkpoint path optional for warm-start scenarios
- Study summary contains Optuna-specific metadata

---

### Component 5: `ValidationCommand`

**Purpose**: Command for validating configuration against strategy requirements with optional dry build to catch runtime mismatches.

**Constructor Parameters**:
- `command_name: str = "validate_config"` - Command identifier

**Key Methods**:
- `validate_input(input_data: ValidationCommandInput, settings: BaseSettingsProtocol) -> None` - Validate command input
- `execute(input_data: ValidationCommandInput, settings: BaseSettingsProtocol, **kwargs: Any) -> bool` - Execute validation
- `_describe_profile(settings: BaseSettingsProtocol) -> str` - Describe configuration profile
- `_optuna_enabled(settings: BaseSettingsProtocol) -> bool` - Check if Optuna enabled
- `_mlflow_enabled(settings: BaseSettingsProtocol) -> bool` - Check if MLflow enabled
- `_is_inference(settings: BaseSettingsProtocol) -> bool` - Check if inference mode

**Input Fields** (`ValidationCommandInput`):
- `dry_build: bool = False` - Perform dry build to catch component instantiation errors

**Returns**: `bool` - True if validation succeeds

**Raises**:
- `WorkflowError` - On validation failure with detailed error message

**Example**:
```python
from dlkit.interfaces.api.commands import ValidationCommand, ValidationCommandInput

command = ValidationCommand()

# Basic validation
input_data = ValidationCommandInput(dry_build=False)
is_valid = command.execute(input_data, settings)

# Deep validation with dry build
input_data = ValidationCommandInput(dry_build=True)
is_valid = command.execute(input_data, settings)  # Catches model/datamodule errors
```

**Implementation Notes**:
- Performs structural validation (required sections present)
- Checks environment dependencies (mlflow, optuna packages)
- Optional dry build instantiates components to catch runtime errors
- Profile description includes mode (training/inference) and enabled features
- Validation errors include profile context for debugging

---

### Component 6: `ConvertCommand`

**Purpose**: Command for converting PyTorch Lightning checkpoints to ONNX format with shape inference and dynamic axes support.

**Constructor Parameters**:
- `command_name: str = "convert"` - Command identifier

**Key Methods**:
- `validate_input(input_data: ConvertCommandInput, settings: BaseSettingsProtocol | None) -> None` - Validate checkpoint and output paths
- `execute(input_data: ConvertCommandInput, settings: BaseSettingsProtocol | None, **kwargs: Any) -> ConvertResult` - Execute conversion
- `_parse_or_infer_shapes(shape_spec: str | None, settings: BaseSettingsProtocol | None, default_batch: int) -> tuple[list[tuple[int, ...]], bool]` - Parse user shapes or infer from dataloader

**Input Fields** (`ConvertCommandInput`):
- `checkpoint_path: Path | str` - Source checkpoint path
- `output_path: Path | str` - Destination ONNX file path
- `shape: str | None = None` - Input shape specification (e.g., "784" or "3,224,224")
- `opset: int = 17` - ONNX opset version
- `batch_size: int | None = None` - Batch dimension for CLI shapes

**Returns**: `ConvertResult` - Contains output path, opset version, input shapes

**Raises**:
- `WorkflowError` - On missing checkpoint, invalid shapes, or conversion failure

**Example**:
```python
from dlkit.interfaces.api.commands import ConvertCommand, ConvertCommandInput

command = ConvertCommand()

# Convert with explicit shape (CLI mode)
input_data = ConvertCommandInput(
    checkpoint_path="model.ckpt",
    output_path="model.onnx",
    shape="784",  # Feature dims (batch added automatically)
    batch_size=1,
    opset=17,
)
result = command.execute(input_data, None)

# Convert with config-based shape inference
input_data = ConvertCommandInput(checkpoint_path="model.ckpt", output_path="model.onnx")
result = command.execute(input_data, settings)  # Infers shape from dataloader
```

**Implementation Notes**:
- Supports both CLI mode (explicit shapes) and config mode (inferred shapes)
- CLI shapes exclude batch dimension (added automatically)
- Config mode gets shapes from first dataloader batch
- Multiple inputs supported via semicolon-separated shapes
- Dynamic axes enable variable batch size in ONNX model
- Wrapper loaded from checkpoint for evaluation mode

---

### Component 7: `CommandDispatcher`

**Purpose**: Dispatcher for routing and executing commands with centralized error handling and logging. Implements Registry pattern for command lookup.

**Constructor Parameters**:
- None - Initializes empty command registry

**Key Attributes**:
- `registry: CommandRegistry` - Registry of command implementations

**Key Methods**:
- `register_command(name: str, command_class: type[BaseCommand[Any, Any]]) -> None` - Register command for execution
- `execute(command_name: str, input_data: Any, settings: BaseSettingsProtocol, **kwargs: Any) -> Any` - Execute registered command
- `list_available_commands() -> list[str]` - List all registered command names

**Returns**: Varies by command - returns command execution result

**Raises**:
- Error raised by `raise_error()` utility on unknown command or execution failure

**Example**:
```python
from dlkit.interfaces.api.commands import CommandDispatcher, TrainCommand

# Create dispatcher and register commands
dispatcher = CommandDispatcher()
dispatcher.register_command("train", TrainCommand)

# Execute command by name
result = dispatcher.execute("train", train_input, settings)

# List available commands
commands = dispatcher.list_available_commands()
print(f"Available: {commands}")

# Global dispatcher instance
from dlkit.interfaces.api.commands import get_dispatcher

dispatcher = get_dispatcher()
```

**Implementation Notes**:
- Maintains global singleton instance via `get_dispatcher()`
- Command lookup by string name enables CLI integration
- Creates fresh command instance for each execution
- Structured logging includes command name, input type, result type
- Error messages include list of available commands on lookup failure

---

### Component 8: `GenerateTemplateCommand` and `ValidateTemplateCommand`

**Purpose**: Commands for generating and validating TOML configuration templates. Support multiple template types (training, inference, mlflow, optuna).

**Template Types** (`TemplateKind`):
- `"training"` - Basic training workflow configuration
- `"inference"` - Inference mode configuration
- `"mlflow"` - Training with MLflow tracking
- `"optuna"` - Hyperparameter optimization with Optuna

**GenerateTemplateCommand**:
- Input: `GenerateTemplateCommandInput(template_type: TemplateKind)`
- Output: `GenerateTemplateCommandOutput(template_content: str, template_type: TemplateKind)`
- Delegates to `ConfigurationService.generate_template()`

**ValidateTemplateCommand**:
- Input: `ValidateTemplateCommandInput(template_content: str, template_type: TemplateKind | None)`
- Output: `ValidateTemplateCommandOutput(is_valid: bool, errors: list[str], template_type: TemplateKind | None)`
- Validates TOML parsing and Settings instantiation

**Example**:
```python
from dlkit.interfaces.api.commands import GenerateTemplateCommand, GenerateTemplateCommandInput

# Generate template
gen_cmd = GenerateTemplateCommand()
gen_input = GenerateTemplateCommandInput(template_type="mlflow")
gen_output = gen_cmd.execute(gen_input, settings)

print(gen_output.template_content)  # TOML string

# Validate template
val_cmd = ValidateTemplateCommand()
val_input = ValidateTemplateCommandInput(
    template_content=gen_output.template_content, template_type="mlflow"
)
val_output = val_cmd.execute(val_input, settings)

if not val_output.is_valid:
    print(f"Errors: {val_output.errors}")
```

**Implementation Notes**:
- Template generation uses Pydantic model introspection
- Validation attempts TOML parsing and Settings instantiation
- Errors collected rather than raised for batch validation
- Templates include helpful comments for user guidance

## Usage Patterns

### Common Use Case 1: Direct Command Execution
```python
from dlkit.interfaces.api.commands import TrainCommand, TrainCommandInput
from dlkit.tools.config import GeneralSettings

# Load settings
settings = GeneralSettings.from_toml("config.toml")

# Create and execute command directly
command = TrainCommand()
input_data = TrainCommandInput(mlflow=True, epochs=100, batch_size=32)

result = command.execute(input_data, settings)
print(f"Training completed: {result.checkpoint_path}")
```

### Common Use Case 2: Dispatcher-Based Execution
```python
from dlkit.interfaces.api.commands import get_dispatcher, TrainCommandInput

# Get global dispatcher
dispatcher = get_dispatcher()

# Execute via dispatcher (CLI pattern)
input_data = TrainCommandInput(epochs=50)
result = dispatcher.execute("train", input_data, settings)
```

### Common Use Case 3: Command Registration and Extension
```python
from dlkit.interfaces.api.commands import BaseCommand, get_dispatcher


# Create custom command
class MyCommand(BaseCommand[MyInput, MyOutput]):
    def validate_input(self, input_data, settings):
        # Validation logic
        pass

    def execute(self, input_data, settings, **kwargs):
        # Execution logic
        return MyOutput(result="done")


# Register with dispatcher
dispatcher = get_dispatcher()
dispatcher.register_command("my_command", MyCommand)

# Execute custom command
result = dispatcher.execute("my_command", my_input, settings)
```

### Common Use Case 4: Validation Before Training
```python
from dlkit.interfaces.api.commands import ValidationCommand, ValidationCommandInput

# Validate configuration before expensive training
val_cmd = ValidationCommand()
val_input = ValidationCommandInput(dry_build=True)

try:
    is_valid = val_cmd.execute(val_input, settings)
    print("Configuration valid - proceeding to training")

    # Proceed with training
    train_cmd = TrainCommand()
    result = train_cmd.execute(train_input, settings)
except WorkflowError as e:
    print(f"Validation failed: {e.message}")
    print(f"Profile: {e.context.get('profile')}")
```

## Error Handling

**Exceptions Raised**:
- `WorkflowError` - On validation failure or execution error (base class for command errors)
- `ConfigurationError` - On template generation or validation failure
- Generic `Exception` - Caught and wrapped in `WorkflowError` with context

**Error Handling Pattern**:
```python
from dlkit.interfaces.api.domain import WorkflowError

try:
    result = command.execute(input_data, settings)
except WorkflowError as e:
    # Structured error with message and context
    logger.error(f"Command failed: {e.message}")
    logger.error(f"Context: {e.context}")

    # Context includes command name, error details
    if "validation_errors" in e.context:
        print(f"Validation errors: {e.context['validation_errors']}")
```

**Error Context**:
- All errors include `command` field identifying the command
- Validation errors include `validation_errors` field with detailed messages
- Execution errors include `error_type` for debugging
- Profile information included in validation errors

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/api/test_commands.py`
- Integration tests: `tests/integration/test_command_integration.py`
- Validation tests: `tests/interfaces/api/test_validation_command.py`

### Key Test Scenarios
1. **Command execution**: Each command type executes successfully with valid input
2. **Input validation**: Invalid inputs raise appropriate WorkflowError
3. **Override application**: Overrides correctly modify settings
4. **Checkpoint handling**: Checkpoint paths validated and loaded properly
5. **Dispatcher routing**: Commands registered and executed via dispatcher
6. **Template generation**: All template types generate valid TOML
7. **Dry build validation**: Component instantiation errors caught

### Fixtures Used
- `general_settings` (from `conftest.py`): Complete settings for command execution
- `tmp_path` (pytest built-in): Temporary paths for checkpoints and output
- `mock_services` (test-specific): Mock services for command isolation

## Performance Considerations
- Commands are stateless - no memory retained between executions
- Settings patched immutably via `BasicSettings.patch()` (small overhead)
- Override validation happens before expensive operations
- Dry build optional for faster validation when not needed
- Command registry uses dict lookup (O(1) complexity)
- Input dataclasses frozen for memory efficiency

## Future Improvements / TODOs
- [ ] Add command execution metrics and timing
- [ ] Support for command chaining (pipeline pattern)
- [ ] Async command execution for long-running workflows
- [ ] Command undo/redo for interactive sessions
- [ ] Batch command execution with parallel processing
- [ ] Command result caching for idempotent operations
- [ ] Rich progress reporting during command execution

## Related Modules
- `dlkit.interfaces.api.services`: Business logic executed by commands
- `dlkit.interfaces.api.domain`: Result types and error hierarchy
- `dlkit.interfaces.api.services.override_service`: Parameter override management
- `dlkit.interfaces.cli`: CLI layer that invokes commands
- `dlkit.tools.config`: Configuration models and loading

## Change Log
- **2025-10-03**: Added comprehensive documentation with examples
- **2024-12-15**: Added ConvertCommand for ONNX export
- **2024-12-01**: Added template generation and validation commands
- **2024-11-20**: Implemented CommandDispatcher with registry pattern
- **2024-11-01**: Initial command implementations (Train, Inference, Optimization)
