# API Interfaces Module

`dlkit.interfaces.api` is a thin external adapter over runtime-owned entrypoints.

## Public Surface
- `train()`
- `optimize()`
- `execute()`
- config/template validation helpers
- MLflow model-registry helpers

Inference remains separate under `dlkit.interfaces.inference`.

## Structure
- `functions/`: public adapter functions that call runtime entrypoints
- `domain/`: API-local TypedDict overrides and small protocol types

## Usage
```python
from dlkit.interfaces.api import execute, optimize, train

training_result = train(settings, overrides={"epochs": 50, "batch_size": 64})
optimization_result = optimize(settings, overrides={"trials": 25, "study_name": "search"})
result = execute(settings, overrides={"run_name": "baseline"})
```

## Design Rule
The API layer stays thin: no workflow orchestration, no command objects, and no
duplicate runtime logic.
