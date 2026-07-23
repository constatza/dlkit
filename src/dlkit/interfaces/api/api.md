# API Interfaces Module

`dlkit.interfaces.api` is a thin external adapter over runtime-owned entrypoints.

## Public Surface
- `train()`
- `optimize()`
- `converge()`
- `execute()`
- `run_multirun_config()` / `run_multirun_spec()` — general multirun sweeps
  (config-driven vs. already-built `MultiRunSpec`)
- config/template validation helpers
- MLflow model-registry helpers

Inference remains separate under `dlkit.interfaces.inference`.

At the package root, `dlkit` keeps a curated flat surface:
- workflows: `train`, `optimize`, `converge`, `execute`
- inference: `load_model`, `evaluate`
- typed config loaders: `load_training_config`, `load_inference_config`, `load_optimization_config`
- registration entrypoints: `register_model`, `register_dataset`

Broader concern-specific surfaces live under:
- `dlkit.config`
- `dlkit.registry`
- `dlkit.inference`

## Structure
- `functions/`: public adapter functions that call runtime entrypoints
- `domain/`: API-local TypedDict overrides and small protocol types

## Usage
```python
from dlkit.interfaces.api import converge, execute, optimize, train
from dlkit.interfaces.api.domain import (
    ConvergenceOverrides,
    ExecutionOverrides,
    OptimizationOverrides,
    TrainingOverrides,
)

training_result = train(settings, overrides=TrainingOverrides(epochs=50, batch_size=64))
optimization_result = optimize(settings, overrides=OptimizationOverrides(trials=25, study_name="search"))
convergence_result = converge(settings, overrides=ConvergenceOverrides(repeats=3))
result = execute(settings, overrides=ExecutionOverrides(run_name="baseline"))

from dlkit.interfaces.api import run_multirun_config

sweep_result = run_multirun_config(multirun_settings)  # MultiRunResult[ChildOutcome[WorkflowResult]]
```

`run_multirun_config()`/`run_multirun_spec()` accept an `mlflow: bool = False`
kwarg for signature symmetry with the other workflow functions, but it has no
effect: a multirun sweep's entire purpose is parent/child MLflow linkage, so
the engine entrypoint always configures tracking from the first child's own
settings regardless of this flag — see
`engine/workflows/entrypoints/entrypoints.md`.

## Design Rule
The API layer stays thin: no workflow orchestration, no command objects, and no
duplicate runtime logic.
