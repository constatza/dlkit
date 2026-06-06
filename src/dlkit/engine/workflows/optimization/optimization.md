# Optimization Module

`dlkit.engine.workflows.optimization` is the runtime-owned subsystem for the
hyperparameter-optimization workflow. The package root is a marker only; import
concrete modules directly.

## Overview

The optimization workflow is a runtime-owned hexagonal subsystem.
Its public runtime entrypoint is `dlkit.engine.workflows.entrypoints.optimize()`.

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

Optimization configuration persistence is opt-in for local files. When an
active tracker is available, small config artifacts should be logged through the
tracking boundary instead of creating implicit durable files on disk.

## Runtime Boundary

Runtime callers should use:
- `dlkit.engine.workflows.entrypoints.optimize()`
- `dlkit.engine.workflows.strategy.OptimizationStrategy`
- concrete imports from `domain`, `application`, or `infrastructure` when needed

The runtime entrypoint is responsible for:
- applying request-level overrides
- managing path context
- creating tracker context when required
- calling the optimization strategy

## Import Rules

- Import concrete modules directly.
- Do not import from `dlkit.engine.workflows.optimization` as a barrel.
- Keep `domain` independent from `application` and `infrastructure`.

## Entry from Runtime

```python
from dlkit.engine.workflows.entrypoints.optimization import optimize

result = optimize(settings, trials=20, study_name="search")
```
