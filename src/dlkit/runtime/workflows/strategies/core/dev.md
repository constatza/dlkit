# Core Training Execution Strategy Module

## Overview
The core module provides the foundational training execution layer for DLKit, implementing pure PyTorch Lightning training workflows following the Single Responsibility Principle (SRP). It defines clean interfaces for training execution and provides a vanilla implementation free from tracking or optimization concerns, enabling composition via the decorator pattern.

## Architecture & Design Patterns
- **Interface Segregation Principle (ISP)**: Separate focused interfaces (`ITrainingExecutor`, `IOptimizationStrategy`) for distinct concerns
- **Single Responsibility Principle (SRP)**: `VanillaExecutor` has one job - execute PyTorch Lightning training
- **Dependency Inversion Principle (DIP)**: Abstract interfaces allow multiple implementations and testing
- **Open/Closed Principle**: Execution can be extended via decorators without modifying core logic
- **Strategy Pattern**: Training execution as pluggable strategy
- **Template Method Pattern**: Common execution flow with customization points

Key architectural decisions:
- Training execution separated from tracking, optimization, and infrastructure concerns
- Direct exception raising via `WorkflowError` - no error wrapper objects
- Components pre-built and injected via `BuildComponents` - executor doesn't construct dependencies
- Metrics and artifacts collected from Lightning trainer after execution
- Best-effort post-training steps (predict, test) with silent failure handling

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `ITrainingExecutor` | Protocol | Abstract training executor interface | N/A |
| `IOptimizationStrategy` | Protocol | Abstract optimization strategy interface | N/A |
| `VanillaExecutor` | Class | Pure Lightning training implementation | N/A |

### Internal Components
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `_collect_metrics` | Function | Convert trainer metrics to plain Python types | `dict[str, Any]` |

### Protocols/Interfaces
| Name | Methods | Purpose |
|------|---------|---------|
| `ITrainingExecutor` | `execute(components, settings)` | Core training execution contract |
| `IOptimizationStrategy` | `execute_optimization(settings)` | Optimization workflow contract |

## Dependencies

### Internal Dependencies
- `dlkit.interfaces.api.domain`: Result objects (`TrainingResult`, `OptimizationResult`) and exceptions (`WorkflowError`)
- `dlkit.tools.config`: Configuration management (`GeneralSettings`)
- `dlkit.runtime.workflows.factories.build_factory`: Component construction (`BuildComponents`)
- `dlkit.interfaces.api.services.precision_service`: Precision configuration service

### External Dependencies
- `pytorch_lightning`: Training framework (`Trainer`, `seed_everything`, `ModelCheckpoint`)
- `pathlib`: Path handling

## Key Components

### Component 1: `ITrainingExecutor`

**Purpose**: Abstract protocol defining the contract for training execution strategies. Enables dependency inversion - workflow orchestrators depend on this abstraction, not concrete executors.

**Methods**:
- `execute(components: BuildComponents, settings: GeneralSettings) -> TrainingResult` - Execute training workflow

**Returns**: `TrainingResult` - Training outcome with metrics, artifacts, and model state

**Raises**:
- `WorkflowError`: If training execution fails

**Example**:
```python
from dlkit.runtime.workflows.strategies.core import ITrainingExecutor, VanillaExecutor
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config import GeneralSettings


# Type-safe executor usage
def train_model(
    executor: ITrainingExecutor, components: BuildComponents, settings: GeneralSettings
):
    result = executor.execute(components, settings)
    print(f"Training completed: {result.metrics}")


# Works with any ITrainingExecutor implementation
executor = VanillaExecutor()
result = train_model(executor, components, settings)
```

**Implementation Notes**:
- Follows Interface Segregation Principle - minimal, focused interface
- Single abstract method keeps interface lean
- Exceptions raised directly for errors - no error wrapper objects
- Components pre-built and injected - executor doesn't handle construction

---

### Component 2: `VanillaExecutor`

**Purpose**: Pure PyTorch Lightning training execution without tracking, optimization, or other orthogonal concerns. Single responsibility: run the training loop and collect results.

**Constructor Parameters**: None - stateless executor

**Key Methods**:
- `execute(components: BuildComponents, settings: GeneralSettings) -> TrainingResult` - Execute pure training workflow

**Returns**: `TrainingResult` with:
- `model_state: None` - State not collected (use checkpoints)
- `metrics: dict[str, Any]` - Aggregated trainer metrics
- `artifacts: dict[str, Path]` - Checkpoint paths from callbacks
- `duration_seconds: float` - Training duration (currently 0.0)

**Raises**:
- `WorkflowError`: If trainer is None or training fails, with traceback in context

**Example**:
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.factories import BuildFactory
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("config.toml")

# Build components
factory = BuildFactory()
components = factory.build_training_components(settings)

# Execute pure training
executor = VanillaExecutor()
result = executor.execute(components, settings)

print(f"Final metrics: {result.metrics}")
print(f"Best checkpoint: {result.artifacts.get('best_checkpoint')}")
print(f"Last checkpoint: {result.artifacts.get('last_checkpoint')}")
```

**Implementation Notes**:
- Sets reproducible seed from `settings.SESSION.seed` before training
- Applies model precision via `ensure_precision_applied()` if available
- Logs precision configuration for debugging
- Executes `trainer.fit()` as core training step
- Post-training `predict()` and `test()` are best-effort (silent failure)
- Metrics collected from multiple trainer sources (callback_metrics, progress_bar_metrics, logged_metrics)
- Checkpoint artifacts extracted from `ModelCheckpoint` callbacks
- Fallback checkpoint discovery via filesystem globbing if callback paths not set
- All exceptions wrapped in `WorkflowError` with full traceback

**Metric Collection Strategy**:
1. Prefer `callback_metrics` - most complete end-of-epoch values
2. Augment with `progress_bar_metrics` and `logged_metrics`
3. Convert all metric values to plain Python float when possible

**Artifact Collection Strategy**:
1. Primary: Extract `best_model_path` and `last_model_path` from `ModelCheckpoint` callbacks
2. Fallback: Search callback `dirpath` for `last.ckpt` or `*-last.ckpt` patterns
3. Fallback: Use any `.ckpt` file in `dirpath` as best checkpoint if nothing else found

---

### Component 3: `IOptimizationStrategy`

**Purpose**: Abstract protocol for optimization workflows that produce `OptimizationResult`. Bridges the gap between training executors (which return `TrainingResult`) and optimization workflows (which return `OptimizationResult` with trial information).

**Methods**:
- `execute_optimization(settings: GeneralSettings) -> OptimizationResult` - Execute optimization workflow

**Returns**: `OptimizationResult` - Optimization outcome with best trial and training result

**Raises**:
- `WorkflowError`: If optimization fails

**Example**:
```python
from dlkit.runtime.workflows.strategies.core import IOptimizationStrategy
from dlkit.tools.config import GeneralSettings


def run_optimization(optimizer: IOptimizationStrategy, settings: GeneralSettings):
    result = optimizer.execute_optimization(settings)
    print(f"Best trial: {result.best_trial}")
    print(f"Best training metrics: {result.training_result.metrics}")


# Works with any IOptimizationStrategy implementation (e.g., OptunaOptimizer)
```

**Implementation Notes**:
- Interface Segregation - focused on optimization concerns only
- Separate from `ITrainingExecutor` to avoid interface pollution
- Enables optimization strategies that internally use training executors
- Return type includes both trial metadata and training result

## Usage Patterns

### Common Use Case 1: Basic Training Execution
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.factories import BuildFactory
from dlkit.tools.config import GeneralSettings

# Load configuration
settings = GeneralSettings.from_toml("training_config.toml")

# Build all required components
factory = BuildFactory()
components = factory.build_training_components(settings)

# Execute training - clean, focused interface
executor = VanillaExecutor()
result = executor.execute(components, settings)

# Access results
print(f"Training metrics: {result.metrics}")
print(f"Validation loss: {result.metrics.get('val_loss', 'N/A')}")

# Access checkpoint artifacts
if best_ckpt := result.artifacts.get("best_checkpoint"):
    print(f"Best model saved at: {best_ckpt}")
```

### Common Use Case 2: Composing with Tracking Decorator
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.runtime.workflows.strategies.tracking import TrackingDecorator, MLflowTracker

# Create base executor
vanilla_executor = VanillaExecutor()

# Wrap with tracking - decorator pattern
tracker = MLflowTracker()
tracked_executor = TrackingDecorator(
    executor=vanilla_executor, tracker=tracker, settings=settings, experiment_name="my_training"
)

# Execute with automatic tracking
result = tracked_executor.execute(components, settings)
# Tracking happens automatically without modifying VanillaExecutor
```

### Common Use Case 3: Custom Executor Implementation
```python
from dlkit.runtime.workflows.strategies.core import ITrainingExecutor
from dlkit.interfaces.api.domain import TrainingResult, WorkflowError


class DistributedExecutor(ITrainingExecutor):
    """Custom executor with distributed training logic."""

    def __init__(self, world_size: int):
        self.world_size = world_size

    def execute(self, components, settings):
        try:
            # Custom distributed setup
            setup_distributed(self.world_size)

            # Reuse standard training flow
            trainer = components.trainer
            trainer.fit(components.model, datamodule=components.datamodule)

            # Custom metric collection
            metrics = collect_distributed_metrics(trainer)

            return TrainingResult(
                model_state=None, metrics=metrics, artifacts={}, duration_seconds=0.0
            )
        except Exception as e:
            raise WorkflowError(f"Distributed training failed: {e}") from e


# Use custom executor - same interface
executor = DistributedExecutor(world_size=4)
result = executor.execute(components, settings)
```

### Common Use Case 4: Testing with Mock Executor
```python
from dlkit.runtime.workflows.strategies.core import ITrainingExecutor
from dlkit.interfaces.api.domain import TrainingResult


class MockExecutor(ITrainingExecutor):
    """Mock executor for testing without actual training."""

    def execute(self, components, settings):
        # Return fake result for testing
        return TrainingResult(
            model_state=None,
            metrics={"val_loss": 0.5, "val_accuracy": 0.95},
            artifacts={},
            duration_seconds=10.0,
        )


# Test workflows without training overhead
def test_workflow_orchestration():
    executor = MockExecutor()
    result = executor.execute(components, settings)
    assert result.metrics["val_accuracy"] > 0.9
```

## Error Handling

**Exceptions Raised**:
- `WorkflowError`: When trainer is None or training execution fails
- All exceptions wrapped with full traceback in error context

**Error Handling Pattern**:
```python
from dlkit.runtime.workflows.strategies.core import VanillaExecutor
from dlkit.interfaces.api.domain import WorkflowError

try:
    executor = VanillaExecutor()
    result = executor.execute(components, settings)
except WorkflowError as e:
    logger.error(f"Training failed: {e}")
    logger.debug(f"Traceback: {e.context.get('trace')}")
    # Handle gracefully - e.g., retry, fallback, alert
```

**Fail-Safe Design**:
- Post-training steps (`predict()`, `test()`) fail silently - don't crash workflow
- Metric collection robust to missing or malformed values
- Checkpoint collection has multiple fallback strategies

## Testing

### Test Coverage
- Unit tests: `tests/runtime/workflows/strategies/test_core_vanilla_executor.py`
- Integration tests: `tests/integration/test_basic_integration.py`
- Wrapper tests: `tests/core/models/wrappers/test_standard_wrapper_steps.py`

### Key Test Scenarios
1. **Basic training flow**: Execute training, verify metrics collected
2. **Checkpoint collection**: Verify best and last checkpoint artifacts
3. **Error handling**: Trainer None raises WorkflowError
4. **Metric aggregation**: Multiple metric sources combined correctly
5. **Precision handling**: Model precision applied before training
6. **Seed reproducibility**: Same seed produces same results
7. **Post-training steps**: Predict/test failures don't crash execution

### Fixtures Used
- `general_settings` (from `conftest.py`): Complete configuration object
- `build_components` (from `conftest.py`): Pre-built training components
- `tmp_path` (pytest built-in): Temporary paths for checkpoints

## Performance Considerations
- Stateless executor design - no state overhead between calls
- Metrics collected once after training, not incrementally
- Checkpoint discovery uses efficient glob patterns
- No unnecessary model state copying (returns None)
- Precision info logged once, not per batch/epoch

## Future Improvements / TODOs
- [ ] Capture training duration accurately
- [ ] Support model state collection in memory (optional)
- [ ] Add validation-only execution mode (skip training)
- [ ] Support resuming from checkpoint
- [ ] Parallel post-training steps (predict + test concurrently)
- [ ] Structured metric schemas for type safety
- [ ] Checkpoint metadata collection (epoch, step, metrics)
- [ ] Progress callbacks for long-running training

## Related Modules
- `dlkit.runtime.workflows.strategies.tracking`: Tracking decorators that wrap executors
- `dlkit.runtime.workflows.strategies.optuna`: Optimization strategies using executors
- `dlkit.runtime.workflows.factories.build_factory`: Component construction for executors
- `dlkit.interfaces.api.domain`: Result objects and exceptions
- `dlkit.interfaces.api.services.precision_service`: Precision configuration service

## Change Log
- **2024-10-03**: Initial core executor implementation with ISP interfaces
- **2024-10-02**: Added precision service integration
- **2024-10-01**: Enhanced checkpoint collection with filesystem fallbacks
- **2024-09-30**: Improved metric collection from multiple trainer sources
- **2024-09-25**: Separated IOptimizationStrategy from ITrainingExecutor
