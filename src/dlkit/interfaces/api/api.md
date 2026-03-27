# API Interfaces Module

## Overview
The `dlkit.interfaces.api` module provides DLKit's public API layer, implementing the Command Pattern with service-based business logic orchestration. It offers a clean, typed interface for training, inference, and optimization workflows while maintaining strict separation of concerns between API, services, and domain layers.

## Module Organization

### Submodules
1. **`commands/`** - Command pattern implementations; includes `normalizer.py` (OverrideNormalizer)
2. **`services/`** - Business logic orchestration; includes `override_service.py` (BasicOverrideManager)
3. **`domain/`** - Error hierarchy, precision, override TypedDicts; result types re-exported from `dlkit.domain`
4. **`functions/`** - Public API functions (entry points)
5. **`adapters/`** - (Empty) Reserved for future adapters

## Architecture Layers

```
┌─────────────────────────────────────────────────┐
│         Public API Functions (Facade)           │
│  train(), optimize(), load_model()             │
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
- `OverrideNormalizer` (`commands/normalizer.py`) — pure utility for path normalization and None-filtering
- `BasicOverrideManager` (`services/override_service.py`) — applies setting patches via `BasicSettings.patch()`
- Override TypedDicts (`domain/override_types.py`) — typed contracts for override inputs
- Thread-local path context (`tools.io.path_context`) — API-level root/output/data dir overrides

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
from dlkit import load_model
import torch

predictor = load_model("model.ckpt", device="auto")
predictions = predictor.predict(x=torch.randn(32, 10))
predictor.unload()
```

### Optimization
```python
from dlkit.interfaces.api import optimize

result = optimize(settings, trials=50, mlflow=True)
print(f"Best trial: {result.best_trial}")
```

## Module Documentation

Detailed documentation for each submodule:
- **Commands**: [`commands/commands.md`](commands/commands.md)
- **Services**: [`services/services.md`](services/services.md)
- **Domain**: [`domain/domain.md`](domain/domain.md)
- **Functions**: [`functions/functions.md`](functions/functions.md)

## Common Patterns

### 1. Override Application
```python
from dlkit.interfaces.api import train

result = train(
    settings,
    epochs=50,  # Training override
    batch_size=64,  # Datamodule override
    output_dir="./exp_1",  # Path override
    mlflow_host="localhost",  # MLflow override
)
```

### 2. Path Context Management
```python
from dlkit.tools.io.path_context import path_override_context, resolve_with_context

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

## Inference

Inference is handled by a dedicated stateful predictor. Use `load_model()` from the
top-level `dlkit` package:

```python
from dlkit import load_model

with load_model("model.ckpt", device="auto") as predictor:
    predictions = predictor.predict(x=input_tensor)
```

See [`interfaces/inference/README.md`](../inference/README.md) for the full inference API.

## Related Documentation
- Architecture Overview: [`AGENTS.md`](../../../../AGENTS.md)
- Inference API: [`interfaces/inference/README.md`](../inference/README.md)
- Configuration Guide: [`tools/config/config.md`](../../tools/config/config.md)

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
