# API Interfaces Module

## Overview
The `dlkit.interfaces.api` module provides DLKit's public API layer, implementing the Command Pattern with service-based business logic orchestration. It offers a clean, typed interface for training, inference, and optimization workflows while maintaining strict separation of concerns between API, services, and domain layers.

## Module Organization

### Submodules
1. **`commands/`** - Command pattern implementations for workflow operations
2. **`services/`** - Business logic orchestration services
3. **`domain/`** - Domain models and error hierarchy
4. **`overrides/`** - Runtime parameter override management
5. **`functions/`** - Public API functions (entry points)
6. **`adapters/`** - (Empty) Reserved for future adapters

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│         Public API Functions (Facade)           │
│  train(), infer(), optimize(), predict()        │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│           Command Dispatcher                     │
│  Routes commands to implementations              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              Commands                            │
│  TrainCommand, InferenceCommand, etc.            │
│  - Input validation                              │
│  - Override application                          │
│  - Service delegation                            │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              Services                            │
│  TrainingService, InferenceService, etc.         │
│  - Workflow orchestration                        │
│  - Resource management                           │
│  - Metric/artifact collection                    │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          Domain Models & Errors                  │
│  TrainingResult, WorkflowError, etc.             │
└─────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Command Pattern
- Each operation encapsulated as a command object
- Uniform execution interface via `execute(input_data, settings)`
- Commands are stateless and composable
- Centralized command registry for dynamic dispatch

### 2. Service Layer
- Services orchestrate business logic without implementing core algorithms
- Each service has single responsibility (training, inference, optimization)
- Services delegate to runtime strategies and factories
- Proper resource lifecycle management with context managers

### 3. Domain-Driven Design
- Domain models represent business concepts (TrainingResult, ModelState)
- Error hierarchy reflects domain concerns (WorkflowError, ConfigurationError)
- Value objects are immutable (`@dataclass(frozen=True)`)
- Clear separation from infrastructure concerns

### 4. Direct Exception Raising
- No error wrapper objects - exceptions raised directly
- Structured error hierarchy with context
- Error context includes correlation IDs for tracing
- CLI layer catches and presents user-friendly messages

### 5. Override Management
- Runtime parameter overrides without mutating settings
- Thread-local path context for API flexibility
- Immutable settings updated via `patch()`
- Override validation before application

## Quick Start Guide

### Training
```python
from dlkit.interfaces.api import train
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml("config.toml")
result = train(settings, mlflow=True, epochs=100, batch_size=32)
print(f"Checkpoint: {result.checkpoint_path}")
```

### Inference (Standalone)
```python
from dlkit.interfaces.api import infer

result = infer(
    checkpoint_path="model.ckpt",
    inputs={"x": torch.randn(32, 10)},
    device="auto"
)
predictions = result.predictions
```

### Optimization
```python
from dlkit.interfaces.api import optimize

result = optimize(settings, trials=50, mlflow=True)
print(f"Best trial: {result.best_trial}")
```

## Module Documentation

Detailed documentation for each submodule:
- **Commands**: `/home/archer/projects/dlkit/src/dlkit/interfaces/api/commands/dev.md`
- **Services**: `/home/archer/projects/dlkit/src/dlkit/interfaces/api/services/dev.md`
- **Domain**: `/home/archer/projects/dlkit/src/dlkit/interfaces/api/domain/dev.md`
- **Overrides**: `/home/archer/projects/dlkit/src/dlkit/interfaces/api/overrides/dev.md`
- **Functions**: `/home/archer/projects/dlkit/src/dlkit/interfaces/api/functions/dev.md`

## Common Patterns

### 1. Override Application
```python
from dlkit.interfaces.api import train

result = train(
    settings,
    epochs=50,              # Training override
    batch_size=64,          # Datamodule override
    output_dir="./exp_1",   # Path override
    mlflow_host="localhost" # MLflow override
)
```

### 2. Path Context Management
```python
from dlkit.interfaces.api.overrides import path_override_context

with path_override_context({"output_dir": "./custom_output"}):
    result = train(settings)  # Uses custom output directory
```

### 3. Error Handling
```python
from dlkit.interfaces.api.domain import WorkflowError

try:
    result = train(settings)
except WorkflowError as e:
    print(f"Error: {e.message}")
    print(f"Context: {e.context}")
```

### 4. Configuration Validation
```python
from dlkit.interfaces.api.commands import ValidationCommand, ValidationCommandInput

val_cmd = ValidationCommand()
val_input = ValidationCommandInput(dry_build=True)

is_valid = val_cmd.execute(val_input, settings)
```

## Testing Strategy

### Test Levels
1. **Unit Tests**: Individual commands, services, and utilities
2. **Integration Tests**: End-to-end API workflows
3. **Contract Tests**: Verify API contracts and type safety

### Key Test Scenarios
- Command execution and validation
- Service orchestration and resource management
- Override application and validation
- Error handling and context propagation
- Path resolution with context overrides

## Future Enhancements

### Planned Features
- [ ] Async API for long-running workflows
- [ ] Streaming results for real-time progress
- [ ] Batch operations for multiple experiments
- [ ] GraphQL API layer
- [ ] REST API adapter
- [ ] gRPC service interface

### Architecture Improvements
- [ ] Command result caching for idempotent operations
- [ ] Service metrics and health checks
- [ ] Distributed workflow coordination
- [ ] Plugin system for custom workflows

## Breaking Changes

### Current Layout
- **Inference API**: `infer()` now requires checkpoint and inputs only (no config)
- **Prediction API**: Use `predict_with_config()` for Lightning-based prediction
- **InferenceWorkflowSettings**: Deprecated - use `TrainingWorkflowSettings` with `SESSION.inference=True`

### Migration Guide
```python
# BEFORE:
from dlkit.tools.config import load_inference_settings
settings = load_inference_settings("config.toml")
result = infer(settings, checkpoint_path)

# NOW:
from dlkit.interfaces.api import infer
result = infer(checkpoint_path, input_data)

# OR for prediction with config:
from dlkit.interfaces.api import predict_with_config
from dlkit.tools.io import load_settings
settings = load_settings("config.toml")
result = predict_with_config(settings, checkpoint_path)
```

## Related Documentation
- Architecture Overview: `/home/archer/projects/dlkit/CLAUDE.md`
- Runtime Workflows: `/home/archer/projects/dlkit/src/dlkit/runtime/workflows/dev.md` (if exists)
- Configuration Guide: `/home/archer/projects/dlkit/src/dlkit/tools/config/dev.md` (if exists)

## Contributors Guide

### Adding a New Command
1. Create command class inheriting from `BaseCommand[TInput, TOutput]`
2. Define frozen dataclass for `TInput`
3. Implement `validate_input()` and `execute()` methods
4. Register command with dispatcher in `__init__.py`
5. Add corresponding service method if needed
6. Write unit and integration tests

### Adding a New Service
1. Create service class with single responsibility
2. Use composition for dependencies (inject in `__init__`)
3. Implement main execution method returning domain result
4. Handle resource lifecycle with context managers
5. Add comprehensive error handling with context
6. Write unit tests with mocked dependencies

## Maintainers
- DLKit Development Team

## Change Log
- **2025-10-03**: Created comprehensive API documentation
- **2024-12-15**: BREAKING CHANGE - New inference API
- **2024-11-20**: Added command dispatcher and registry
- **2024-11-01**: Initial API layer implementation
