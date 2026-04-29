# API Interfaces Module

`dlkit.interfaces.api` is a thin external adapter over runtime-owned entrypoints.

## Public Surface
- `train()`
- `optimize()`
- `execute()`
- config/template validation helpers
- MLflow model-registry helpers

Inference remains separate under `dlkit.interfaces.inference`.

At the package root, `dlkit` keeps a curated flat surface:
- workflows: `train`, `optimize`, `execute`
- inference: `load_model`
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
from dlkit.interfaces.api import execute, optimize, train
from dlkit.interfaces.api.domain import TrainingOverrides, OptimizationOverrides, ExecutionOverrides

training_result = train(settings, overrides=TrainingOverrides(epochs=50, batch_size=64))
optimization_result = optimize(settings, overrides=OptimizationOverrides(trials=25, study_name="search"))
result = execute(settings, overrides=ExecutionOverrides(run_name="baseline"))
```

## Design Rule
The API layer stays thin: no workflow orchestration, no command objects, and no
duplicate runtime logic.
