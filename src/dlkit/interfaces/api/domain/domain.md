# Domain Module

## Overview
The domain module defines core domain models and error hierarchy for DLKit's API layer. It provides typed result objects for workflows (training, inference, optimization) and a structured exception hierarchy following domain-driven design principles.

## Architecture & Design Patterns
- **Domain-Driven Design**: Domain models represent business concepts, not technical artifacts
- **Value Objects**: Immutable frozen dataclasses for results
- **Error Hierarchy**: Structured exceptions with context for debugging
- **Type Safety**: Full type hints for compile-time checking
- **Separation of Concerns**: Domain models independent of infrastructure

## Module Structure

### Public API
| Name | Type | Purpose | Returns |
|------|------|---------|---------|
| `TrainingResult` | Dataclass | Training workflow result | N/A |
| `InferenceResult` | Dataclass | Inference workflow result | N/A |
| `OptimizationResult` | Dataclass | Optimization workflow result | N/A |
| `ModelState` | Dataclass | Complete model and component state | N/A |
| `DLKitError` | Exception | Base exception for all domain errors | N/A |
| `ConfigurationError` | Exception | Configuration validation or loading error | N/A |
| `WorkflowError` | Exception | Workflow execution error | N/A |
| `StrategyError` | Exception | Strategy selection or execution error | N/A |
| `ModelStateError` | Exception | Model state construction or management error | N/A |
| `PluginError` | Exception | Plugin configuration or execution error | N/A |

## Key Components

### Component 1: Result Models

**TrainingResult**:
```python
@dataclass(frozen=True)
class TrainingResult:
    model_state: ModelState
    metrics: dict[str, Any]
    artifacts: dict[str, Path]
    duration_seconds: float

    @property
    def checkpoint_path(self) -> Path | None:
        """Get best or last checkpoint path."""
```

**InferenceResult**:
```python
@dataclass(frozen=True)
class InferenceResult:
    model_state: ModelState
    predictions: Any
    metrics: dict[str, Any] | None
    duration_seconds: float
```

**OptimizationResult**:
```python
@dataclass(frozen=True)
class OptimizationResult:
    best_trial: Any
    training_result: TrainingResult
    study_summary: dict[str, Any]
    duration_seconds: float
```

**ModelState**:
```python
@dataclass(frozen=True)
class ModelState:
    model: LightningModule
    datamodule: LightningDataModule
    trainer: Any | None
    settings: GeneralSettings
```

### Component 2: Error Hierarchy

**DLKitError** (base):
```python
class DLKitError(Exception):
    def __init__(self, message: str, context: dict[str, str] | None = None):
        self.message = message
        self.context = context or {}

    @property
    def correlation_id(self) -> str | None:
        return self.context.get("correlation_id")
```

**Specialized Errors**:
- `ConfigurationError` - Config validation/loading failures
- `WorkflowError` - Workflow execution failures
- `StrategyError` - Strategy selection/execution failures
- `ModelStateError` - Model state construction failures
- `PluginError` - Plugin configuration/execution failures

## Usage Patterns

### Working with Results
```python
# Training result
result = train(settings)
print(f"Duration: {result.duration_seconds}s")
print(f"Metrics: {result.metrics}")
checkpoint = result.checkpoint_path  # Property for convenience

# Inference result
result = infer(checkpoint_path, inputs)
predictions = result.predictions

# Optimization result
result = optimize(settings, trials=50)
print(f"Best trial: {result.best_trial}")
print(f"Study summary: {result.study_summary}")
```

### Error Handling
```python
from dlkit.interfaces.api.domain import WorkflowError, ConfigurationError

try:
    result = train(settings)
except ConfigurationError as e:
    print(f"Config error: {e.message}")
    print(f"Context: {e.context}")
except WorkflowError as e:
    print(f"Workflow failed: {e.message}")
    if e.correlation_id:
        print(f"Correlation ID: {e.correlation_id}")
```

## Related Modules
- `dlkit.interfaces.api.commands`: Commands return domain result types
- `dlkit.interfaces.api.services`: Services create and populate results
- `dlkit.tools.config`: Settings used in ModelState

## Change Log
- **2025-10-03**: Added comprehensive documentation
- **2024-11-01**: Initial domain models and error hierarchy
