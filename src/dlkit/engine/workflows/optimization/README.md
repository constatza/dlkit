# Optimization Module

`dlkit.engine.workflows.optimization` is an internal runtime package for the
hyperparameter optimization workflow.

The package root is a marker only. Import concrete modules directly.

## Layout
- `domain/`: optimization entities and protocols
- `application/`: orchestration services
- `infrastructure/`: Optuna and MLflow adapters
- `factory.py`: runtime optimization factory
- `strategy.py`: runtime-facing optimization strategy

## Layering
- `domain` does not depend on `application` or `infrastructure`
- `application` depends on `domain`
- `infrastructure` depends on `domain`

## Usage
Runtime callers should use:
- `dlkit.engine.workflows.entrypoints.optimize()`
- `dlkit.engine.workflows.strategy.OptimizationStrategy`
- concrete imports from `domain`, `application`, or `infrastructure` when needed
