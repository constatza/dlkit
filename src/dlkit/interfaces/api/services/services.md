# Services Module

## Overview
The services module provides business logic orchestration for DLKit workflows, acting as the middle layer between commands (API requests) and runtime execution strategies. Services coordinate component building, resource management, and workflow execution while maintaining separation of concerns and clean dependency boundaries.

## Architecture & Design Patterns
- **Service Layer Pattern**: Business logic separated from API and domain layers
- **Orchestration**: Services coordinate multiple components without implementing core logic
- **Dependency Inversion**: Services depend on abstractions (Orchestrator, BuildFactory) not implementations
- **Resource Management**: Proper lifecycle handling with context managers
- **Path Context Integration**: Thread-local path overrides for API flexibility
- **Fail-Fast Validation**: Early validation before expensive operations

Key architectural decisions:
- Services own workflow orchestration, not domain logic
- Each service has single responsibility (training, inference, optimization, configuration, precision)
- Services use Orchestrator and BuildFactory for component coordination
- Path overrides applied via thread-local context to avoid global pollution
- Timing and metrics collection at service boundary

## Module Structure

### Public API
| Name | File | Type | Purpose | Returns |
|------|------|------|---------|---------|
| `TrainingService` | `training_service.py` | Class | Orchestrate training workflows | `TrainingResult` |
| `InferenceService` | `inference_service.py` | Class | Orchestrate inference workflows | `InferenceResult` |
| `OptimizationService` | `optimization_service.py` | Class | Orchestrate Optuna optimization workflows | `OptimizationResult` |
| `ExecutionService` | `execution_service.py` | Class | Intelligent workflow routing based on settings | `TrainingResult \| InferenceResult \| OptimizationResult` |
| `ConfigurationService` | `configuration_service.py` | Class | Generate and validate configuration templates | `str` or validation results |
| `PrecisionService` | `precision_service.py` | Class | Centralized precision coordination and resolution | Various precision types |
| `get_precision_service()` | `precision_service.py` | Function | Get global precision service instance | `PrecisionService` |
| `BasicOverrideManager` | `override_service.py` | Class | Apply runtime parameter overrides to settings | Updated settings |
| `basic_override_manager` | `override_service.py` | Instance | Global singleton override manager | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `WorkflowDetectionResult` | Dataclass | Workflow detection result | N/A |
| `_extract_metrics()` | Method | Extract training metrics from model state | `dict[str, Any]` |
| `_collect_artifacts()` | Method | Collect training artifacts (checkpoints, logs) | `dict[str, Path]` |
| `_run_inference()` | Method | Execute inference with model and datamodule | `Any` |
| `_detect_workflow()` | Method | Detect workflow type from settings | `WorkflowDetectionResult` |
| `_build_template_dict()` | Method | Build template dictionary from Pydantic models | `dict[str, Any]` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| N/A | N/A | Services use concrete classes and existing protocols |

## Dependencies

### Internal Dependencies
- `dlkit.runtime.workflows.orchestrator`: Main workflow orchestrator
- `dlkit.runtime.workflows.factories.build_factory`: Component building
- `dlkit.domain`: Shared result types (`TrainingResult`, `InferenceResult`, `OptimizationResult`, `ModelState`)
- `dlkit.interfaces.api.domain`: Error types and precision
- `dlkit.interfaces.api.commands.normalizer`: `OverrideNormalizer` for path normalization
- `dlkit.tools.io.path_context`: Thread-local path overrides (`path_override_context`, `resolve_with_context`)
- `dlkit.tools.config`: Configuration models and settings
- `dlkit.tools.io`: Path resolution and provisioning
- `dlkit.interfaces.servers.domain_functions`: Server configuration helpers

### External Dependencies
- `torch`: Model loading and tensor operations
- `lightning.pytorch`: Training framework integration
- `tomlkit`: TOML template generation
- `pydantic`: Configuration model introspection

## Key Components

### Component 1: `TrainingService`

**Purpose**: Orchestrate training workflows via Orchestrator, handling component building, execution, metric extraction, and artifact collection. Coordinates resource provisioning and path override contexts.

**Constructor Parameters**:
- None - Initializes service with default orchestrator

**Key Attributes**:
- `service_name: str = "training_service"` - Service identifier for logging

**Key Methods**:
- `execute_training(settings: GeneralSettings, checkpoint_path: Path | None = None) -> TrainingResult` - Execute training workflow
- `_extract_metrics(model_state) -> dict[str, Any]` - Extract metrics from trainer callbacks
- `_collect_artifacts(model_state) -> dict[str, Path]` - Collect checkpoints and log directories

**Parameters** (`execute_training`):
- `settings: GeneralSettings` - DLKit configuration settings
- `checkpoint_path: Path | None = None` - Optional checkpoint for resuming training

**Returns**: `TrainingResult` with model state, metrics, artifacts, duration

**Raises**:
- `WorkflowError` - On execution failure with service context

**Example**:
```python
from dlkit.interfaces.api.services import TrainingService
from dlkit.tools.config import GeneralSettings

# Initialize service
service = TrainingService()

# Load settings
settings = GeneralSettings.from_toml("config.toml")

# Execute training
result = service.execute_training(settings)

# Access results
print(f"Duration: {result.duration_seconds:.2f}s")
print(f"Best checkpoint: {result.checkpoint_path}")
print(f"Metrics: {result.metrics}")

# Resume from checkpoint
result = service.execute_training(settings, checkpoint_path=result.checkpoint_path)
```

**Implementation Notes**:
- Ensures run directories exist via `provisioning.ensure_run_dirs()`
- Applies settings-defined root_dir via path context if not already overridden
- Uses Orchestrator for component building and execution
- Measures duration at service boundary for accurate timing
- Extracts metrics from trainer callbacks (callback_metrics, progress_bar_metrics, logged_metrics)
- Collects checkpoints from ModelCheckpoint callbacks with filesystem fallback
- Logs collected under standardized output policy via `locations.output("logs")`

---

### Component 2: `InferenceService`

**Purpose**: Orchestrate inference workflows via BuildFactory, handling checkpoint loading, component building, and prediction execution. Supports both trainer-based and lightweight inference modes.

**Constructor Parameters**:
- None - Initializes service with default build factory

**Key Attributes**:
- `service_name: str = "inference_service"` - Service identifier for logging

**Key Methods**:
- `execute_inference(settings: GeneralSettings, checkpoint_path: Path) -> InferenceResult` - Execute inference workflow
- `_run_inference(components) -> Any` - Run actual inference using model and datamodule

**Parameters** (`execute_inference`):
- `settings: GeneralSettings` - DLKit configuration settings
- `checkpoint_path: Path` - Path to model checkpoint

**Returns**: `InferenceResult` with predictions, metrics, duration

**Raises**:
- `WorkflowError` - On checkpoint loading failure or execution error

**Example**:
```python
from dlkit.interfaces.api.services import InferenceService
from pathlib import Path

# Initialize service
service = InferenceService()

# Execute inference
result = service.execute_inference(settings, checkpoint_path=Path("./checkpoints/best.ckpt"))

# Access predictions
predictions = result.predictions
print(f"Duration: {result.duration_seconds:.2f}s")
```

**Implementation Notes**:
- Applies settings-defined root_dir via path context if not already overridden
- Uses BuildFactory to build components (trainer is None in inference mode)
- Loads checkpoint with `weights_only=False` for trusted dlkit checkpoints
- Supports both plain state_dict and Lightning-style {'state_dict': ...} format
- Best-effort load with `strict=False` to handle partial checkpoints
- Fallback strips 'model.' prefix if initial load fails
- Uses trainer.predict() if trainer available, otherwise creates lightweight trainer
- Returns empty predictions list if no datamodule/predict_dataloader available

---

### Component 3: `OptimizationService`

**Purpose**: Orchestrate Optuna hyperparameter optimization workflows with experiment tracker lifecycle management. Coordinates optimization strategy selection and execution with proper resource cleanup.

**Constructor Parameters**:
- None - Initializes service with default orchestrator

**Key Attributes**:
- `service_name: str = "optimization_service"` - Service identifier for logging

**Key Methods**:
- `execute_optimization(settings: GeneralSettings, trials: int = 100, checkpoint_path: Path | None = None) -> OptimizationResult` - Execute optimization workflow
- `get_optimization_progress(study_name: str) -> dict[str, Any]` - Get progress for ongoing study (placeholder)

**Parameters** (`execute_optimization`):
- `settings: GeneralSettings` - DLKit configuration settings
- `trials: int = 100` - Number of optimization trials to run
- `checkpoint_path: Path | None = None` - Optional warm-start checkpoint

**Returns**: `OptimizationResult` with best trial, training result, study summary, duration

**Raises**:
- `WorkflowError` - On execution failure with service context

**Example**:
```python
from dlkit.interfaces.api.services import OptimizationService

# Initialize service
service = OptimizationService()

# Execute optimization
result = service.execute_optimization(settings, trials=50)

# Access results
print(f"Best trial: {result.best_trial}")
print(f"Best parameters: {result.study_summary}")
print(f"Training result: {result.training_result}")
```

**Implementation Notes**:
- Applies settings-defined root_dir via path context if not already overridden
- Creates orchestrator and selects optimization strategy
- Detects CleanOptimizationStrategy vs legacy strategy
- Service layer manages experiment tracker context lifecycle
- Tracker context entered before optimization, exited after completion
- Path override context and tracker context properly nested
- Duration measured at service boundary
- Progress tracking placeholder for future Optuna dashboard integration

---

### Component 4: `ExecutionService`

**Purpose**: Intelligent workflow routing service that automatically determines execution path (training, inference, optimization) based on settings. Eliminates need for explicit workflow selection in simple use cases.

**Constructor Parameters**:
- None - Initializes with all underlying services

**Key Attributes**:
- `training_service: TrainingService` - Training workflow service
- `inference_service: InferenceService` - Inference workflow service
- `optimization_service: OptimizationService` - Optimization workflow service
- `service_name: str = "execution_service"` - Service identifier

**Key Methods**:
- `execute(settings: GeneralSettings, ...) -> TrainingResult | InferenceResult | OptimizationResult` - Execute workflow with intelligent routing
- `_detect_workflow(settings: GeneralSettings, checkpoint_path: Path | str | None, mlflow_override: bool) -> WorkflowDetectionResult` - Detect workflow type
- `_is_mlflow_enabled(settings: GeneralSettings) -> bool` - Check MLflow enabled
- `_is_optuna_enabled(settings: GeneralSettings) -> bool` - Check Optuna enabled
- `_execute_training(...)` - Execute training with overrides
- `_execute_inference(...)` - Execute inference with overrides
- `_execute_optimization(...)` - Execute optimization with overrides

**Workflow Detection Priority**:
1. **Inference mode** (highest) - When `settings.SESSION.inference=True`
2. **Optimization mode** - When `settings.OPTUNA.enabled=True`
3. **Training mode** (default) - All other cases

**Parameters** (`execute`):
- `settings: GeneralSettings` - DLKit configuration
- `mlflow: bool = False` - Enable MLflow tracking
- `checkpoint_path: Path | str | None = None` - Checkpoint path (may indicate inference)
- `root_dir: Path | str | None = None` - Override root directory
- `output_dir: Path | str | None = None` - Override output directory
- `data_dir: Path | str | None = None` - Override data directory
- `epochs: int | None = None` - Override training epochs
- `batch_size: int | None = None` - Override batch size
- `learning_rate: float | None = None` - Override learning rate
- `trials: int | None = None` - Override optimization trials
- `study_name: str | None = None` - Override Optuna study name
- `experiment_name: str | None = None` - Override MLflow experiment name
- `run_name: str | None = None` - Override MLflow run name
- `**additional_overrides: Any` - Extra parameter overrides

**Returns**: Result type based on detected workflow

**Raises**:
- `WorkflowError` - On detection failure or execution error

**Example**:
```python
from dlkit.interfaces.api.services import ExecutionService

# Initialize service
service = ExecutionService()

# Automatic workflow detection and execution
result = service.execute(settings, mlflow=True, epochs=50)

# Service automatically routes to:
# - InferenceService if settings.SESSION.inference=True
# - OptimizationService if settings.OPTUNA.enabled=True
# - TrainingService otherwise

# Check result type
if isinstance(result, TrainingResult):
    print(f"Training completed: {result.checkpoint_path}")
elif isinstance(result, OptimizationResult):
    print(f"Best trial: {result.best_trial}")
elif isinstance(result, InferenceResult):
    print(f"Predictions: {result.predictions}")
```

**Implementation Notes**:
- Explicit workflow detection with logging of reasoning
- Override manager applies overrides before service execution
- Inference requires checkpoint_path parameter
- Optimization defaults to 100 trials if not specified
- Path overrides converted to Path objects automatically
- WorkflowDetectionResult includes profile information for debugging
- All execution methods validate required parameters

---

### Component 5: `ConfigurationService`

**Purpose**: Service for configuration template generation and validation following SOLID principles. Introspects Pydantic models to generate TOML templates rather than using static dictionaries.

**Constructor Parameters**:
- None - Stateless service with class methods

**Key Methods**:
- `generate_template(template_type: TemplateKind) -> str` - Generate TOML template
- `_build_template_dict(template_type: TemplateKind) -> dict[str, Any]` - Build template dictionary
- `_extract_model_fields(model_class: type[BaseModel]) -> dict[str, Any]` - Extract fields from Pydantic model
- `_generate_placeholder_value(field_info: FieldInfo) -> Any` - Generate type-appropriate placeholders
- `_customize_for_training(base_dict: dict) -> dict` - Customize for training workflow
- `_customize_for_inference(base_dict: dict) -> dict` - Customize for inference workflow
- `_customize_for_mlflow(base_dict: dict) -> dict` - Customize for MLflow tracking
- `_customize_for_optuna(base_dict: dict) -> dict` - Customize for Optuna optimization
- `_get_field_comments(template_type: TemplateKind) -> dict[str, str]` - Get field comments
- `_render_toml(template: dict, *, kind: TemplateKind) -> str` - Render template as TOML with comments

**Template Types** (`TemplateKind`):
- `"training"` - Basic training workflow
- `"inference"` - Inference mode
- `"mlflow"` - Training with MLflow
- `"optuna"` - Hyperparameter optimization

**Returns**: `str` - TOML configuration template with comments

**Raises**:
- `ValueError` - On unknown template type

**Example**:
```python
from dlkit.interfaces.api.services import ConfigurationService

# Generate training template
template = ConfigurationService.generate_template("training")
print(template)
# [SESSION]
# name = "my_session"  # Human-readable run/session name
# ...

# Generate MLflow template
template = ConfigurationService.generate_template("mlflow")

# Generate Optuna template
template = ConfigurationService.generate_template("optuna")
```

**Implementation Notes**:
- Builds minimal essential sections manually to avoid model complexity
- Skips complex model fields with `extra='allowed'` that have dynamic fields
- Generates type-appropriate placeholder values (no None in TOML)
- Handles Optional types by unwrapping Union[X, None]
- Templates include free-form EXTRAS section for user-defined values
- Deterministic section ordering for consistent output
- Paths resolve relative to DLKIT_ROOT_DIR or CWD (no PATHS section)
- Comments provide helpful guidance for each field
- Uses tomlkit for comment-preserving TOML generation

---

### Component 6: `PrecisionService`

**Purpose**: Centralized precision coordination service that manages precision strategy resolution across all DLKit components. Acts as single source of truth for precision-related decisions.

**Constructor Parameters**:
- `context: PrecisionContext | None = None` - Precision context (uses global if None)

**Key Attributes**:
- `_context: PrecisionContext` - Precision context for override management

**Key Methods**:
- `resolve_precision(provider: PrecisionProvider | None = None, default: PrecisionStrategy | None = None) -> PrecisionStrategy` - Resolve effective precision
- `get_torch_dtype(provider, default) -> torch.dtype` - Get torch.dtype for resolved precision
- `get_compute_dtype(provider, default) -> torch.dtype` - Get computation dtype (handles mixed precision)
- `get_lightning_precision(provider, default) -> str | int` - Get Lightning Trainer precision parameter
- `is_mixed_precision(provider, default) -> bool` - Check if using automatic mixed precision
- `validate_precision_compatibility(precision: PrecisionStrategy, device_type: str = "cuda") -> bool` - Validate precision compatibility
- `cast_tensor(tensor: torch.Tensor, provider, default) -> torch.Tensor` - Cast tensor to precision dtype
- `apply_precision_to_model(model: torch.nn.Module, provider, default) -> torch.nn.Module` - Apply precision to model weights
- `get_precision_info(provider, default) -> dict[str, Any]` - Get comprehensive precision information

**Resolution Priority**:
1. Context override (thread-local API overrides)
2. Provider precision (component-specific configuration)
3. Explicit default parameter
4. Global default (FULL_32)

**Returns**: Varies by method - precision strategies, dtypes, or configuration values

**Example**:
```python
from dlkit.interfaces.api.services import get_precision_service
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.tools.config.precision.strategy import PrecisionStrategy

# Get global service
service = get_precision_service()

# Resolve precision from settings
precision = service.resolve_precision(provider=settings.SESSION)

# Get torch dtype
dtype = service.get_torch_dtype(provider=settings.SESSION)

# Get Lightning precision config
lightning_precision = service.get_lightning_precision(provider=settings.SESSION)

# Use context override
with precision_override(PrecisionStrategy.MIXED_16):
    precision = service.resolve_precision()
    assert precision == PrecisionStrategy.MIXED_16

# Cast tensor to precision
tensor = torch.randn(10, 10)
casted = service.cast_tensor(tensor, provider=settings.SESSION)

# Apply precision to model
model = MyModel()
model = service.apply_precision_to_model(model, provider=settings.SESSION)

# Get precision info for debugging
info = service.get_precision_info(provider=settings.SESSION)
print(f"Strategy: {info['strategy']}")
print(f"Torch dtype: {info['torch_dtype']}")
```

**Implementation Notes**:
- Depends on PrecisionProvider protocol, not concrete classes (DIP)
- Thread-local context enables API overrides without global pollution
- Graceful fallback if provider doesn't support precision
- Mixed precision returns float32 for compute dtype (gradients)
- Validation placeholder for future device/version compatibility checks
- Comprehensive info method for debugging and logging
- Global singleton instance via `get_precision_service()`

## Usage Patterns

### Common Use Case 1: Direct Service Execution
```python
from dlkit.interfaces.api.services import TrainingService
from dlkit.tools.config import GeneralSettings

# Load settings and execute training
settings = GeneralSettings.from_toml("config.toml")
service = TrainingService()
result = service.execute_training(settings)

# Access results
print(f"Metrics: {result.metrics}")
print(f"Checkpoint: {result.checkpoint_path}")
```

### Common Use Case 2: Intelligent Workflow Routing
```python
from dlkit.interfaces.api.services import ExecutionService

# Single service handles all workflow types
service = ExecutionService()

# Automatically routes based on settings
result = service.execute(settings, mlflow=True, epochs=50)

# Result type determined by configuration
# - TrainingResult if normal training
# - OptimizationResult if OPTUNA enabled
# - InferenceResult if SESSION.inference=True
```

### Common Use Case 3: Template Generation
```python
from dlkit.interfaces.api.services import ConfigurationService

# Generate template for specific workflow
template = ConfigurationService.generate_template("mlflow")

# Write to file
with open("config.toml", "w") as f:
    f.write(template)

# Generate all templates
for template_type in ["training", "inference", "mlflow", "optuna"]:
    template = ConfigurationService.generate_template(template_type)
    with open(f"{template_type}_config.toml", "w") as f:
        f.write(template)
```

### Common Use Case 4: Precision Coordination
```python
from dlkit.interfaces.api.services import get_precision_service
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.tools.config.precision.strategy import PrecisionStrategy

service = get_precision_service()

# Normal resolution from settings
precision = service.resolve_precision(provider=settings.SESSION)
dtype = service.get_torch_dtype(provider=settings.SESSION)

# API override for experimentation
with precision_override(PrecisionStrategy.MIXED_16):
    # All components use MIXED_16 during this context
    model = build_model()  # Uses MIXED_16
    result = train(model)  # Uses MIXED_16

# Back to normal precision after context exit
```

## Error Handling

**Exceptions Raised**:
- `WorkflowError` - On execution failure (training, inference, optimization)
- `ValueError` - On invalid template type in ConfigurationService
- Generic `Exception` - Caught and wrapped in WorkflowError with service context

**Error Handling Pattern**:
```python
from dlkit.interfaces.api.domain import WorkflowError

try:
    result = service.execute_training(settings)
except WorkflowError as e:
    logger.error(f"Service failed: {e.message}")
    logger.error(f"Service: {e.context.get('service')}")
    logger.error(f"Error: {e.context.get('error')}")
```

**Service Error Context**:
- All errors include `service` field identifying the service
- Execution errors include `error` field with exception message
- Checkpoint errors include `checkpoint` field with path
- Workflow detection failures include `workflow` field

## Testing

### Test Coverage
- Unit tests: `tests/interfaces/api/test_services.py`
- Integration tests: `tests/integration/test_service_integration.py`
- Precision tests: `tests/tools/config/precision/test_precision_service.py`

### Key Test Scenarios
1. **Service execution**: Each service executes successfully with valid settings
2. **Metric extraction**: Training metrics correctly extracted from callbacks
3. **Artifact collection**: Checkpoints and logs properly collected
4. **Checkpoint loading**: Inference loads checkpoints with fallback handling
5. **Workflow detection**: ExecutionService correctly routes to underlying services
6. **Template generation**: All template types generate valid parseable TOML
7. **Precision resolution**: Correct priority order for precision sources

### Fixtures Used
- `general_settings` (from `conftest.py`): Complete settings for service execution
- `tmp_path` (pytest built-in): Temporary paths for checkpoints and output
- `mock_orchestrator` (test-specific): Mock orchestrator for service isolation
- `precision_context` (test-specific): Precision context for override testing

## Performance Considerations
- Services are stateless - no memory retained between calls
- Path context uses thread-local storage (minimal overhead)
- Metric extraction uses dict comprehension for efficiency
- Checkpoint collection checks filesystem only as fallback
- Template generation caches field comments for reuse
- Precision resolution uses priority short-circuit evaluation
- Orchestrator instantiation deferred until needed

## Future Improvements / TODOs
- [ ] Add service-level caching for idempotent operations
- [ ] Support async service execution for long-running workflows
- [ ] Add service metrics collection and export
- [ ] Implement retry logic for transient failures
- [ ] Add service health checks and readiness probes
- [ ] Support distributed training coordination
- [ ] Add service-level rate limiting and quotas

## Related Modules
- `dlkit.interfaces.api.commands`: Commands that invoke services
- `dlkit.interfaces.api.domain`: Result types and error hierarchy
- `dlkit.runtime.workflows.orchestrator`: Workflow execution coordinator
- `dlkit.runtime.workflows.factories.build_factory`: Component building
- `dlkit.tools.config`: Configuration models and settings
- `dlkit.tools.io`: Path resolution and provisioning

## Change Log
- **2025-10-03**: Added comprehensive documentation with examples
- **2024-12-20**: Added PrecisionService for centralized precision coordination
- **2024-12-10**: Added ConfigurationService for template generation
- **2024-11-25**: Added ExecutionService for intelligent workflow routing
- **2024-11-15**: Added path context integration for API overrides
- **2024-11-01**: Initial service implementations (Training, Inference, Optimization)
