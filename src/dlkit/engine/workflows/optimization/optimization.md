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
- study repository, backend-session, and tracking protocols

### Application

Orchestration services that coordinate:
- study lifecycle
- optimization backend-session lifecycle
- trial execution
- configuration preparation
- interaction with runtime build components

### Infrastructure

Adapters for external systems:
- Optuna persistence
- Optuna backend sessions
- shared backend-study registry for Optuna repository/session internals
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

The optimization orchestrator is responsible for:
- entering and exiting `IOptimizationBackendSession`
- coordinating backend-specific sampling and reporting through that session
- entering tracker-owned nested run contexts after the backend session is active

The runtime entrypoint is responsible for:
- applying request-level overrides
- managing path context
- entering and exiting the top-level experiment tracker when tracking is enabled
- calling the optimization strategy

The factory is responsible for:
- creating `IStudyRepository`, `IOptimizationBackendSession`, trackers, and persisters
- wiring the shared backend-study registry only when `OPTUNA.enabled` is true
- returning unentered context-manager dependencies

## Import Rules

- Import concrete modules directly.
- Do not import from `dlkit.engine.workflows.optimization` as a barrel.
- Keep `domain` independent from `application` and `infrastructure`.
- Keep `IStudyRepository` backend-agnostic; backend-branded operations belong on
  `IOptimizationBackendSession`.
- Keep backend-study resolution in infrastructure internals; repositories do not
  expose backend-native Optuna objects as a consumed contract.

## Lifecycle Guarantees

- Backend sessions must not leave active Optuna trial handles after context exit.
- Sampling failures must finalize or discard backend trial state before re-raising.

## Entry from Runtime

```python
from dlkit.engine.workflows.entrypoints.optimization import optimize

result = optimize(settings, trials=20, study_name="search")
```
