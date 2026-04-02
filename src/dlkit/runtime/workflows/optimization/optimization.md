# Optimization Architecture

## Overview
The optimization workflow is a runtime-owned hexagonal subsystem.
Its public runtime entrypoint is `dlkit.runtime.workflows.entrypoints.optimize()`.

## Internal Layers

### Domain
Pure optimization concepts and contracts:
- study and trial models
- optimization result model
- repository and tracking protocols

### Application
Orchestration services that coordinate:
- study lifecycle
- trial execution
- configuration preparation
- interaction with runtime build components

### Infrastructure
Adapters for external systems:
- Optuna persistence
- MLflow tracking
- configuration serialization

## Import Rules
- Import concrete modules directly.
- Do not import from `dlkit.runtime.workflows.optimization` as a barrel.
- Keep `domain` independent from `application` and `infrastructure`.

## Entry from Runtime

```python
from dlkit.runtime.workflows.entrypoints.optimization import optimize

result = optimize(settings, trials=20, study_name="search")
```

## Runtime Boundary
The runtime entrypoint is responsible for:
- applying request-level overrides
- managing path context
- creating tracker context when required
- calling the optimization strategy
