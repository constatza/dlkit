# API Functions Module

`dlkit.interfaces.api.functions` is a façade over runtime-owned entrypoints.

## Responsibilities
- expose the public workflow functions
- accept strict Pydantic override payloads (`TrainingOverrides`, `OptimizationOverrides`, `ConvergenceOverrides`, `ExecutionOverrides`)
- coerce known string paths to `Path`
- forward normalized requests to runtime entrypoints

## Public Functions
- `train()`
- `optimize()`
- `converge()`
- `execute()`
- `validate_config()`
- `generate_template()`
- `validate_template()`
- logged-model helpers: `search_logged_models()`, `build_logged_model_uri()`, `load_logged_model()`
- registry helpers: `register_logged_model()`, `search_registered_models()`, `list_model_versions()`, `get_model_version()`, `set_registered_model_alias()`, `set_registered_model_version_tag()`, `set_registered_model_version_tags()`, `build_registered_model_uri()`, `load_registered_model()`
- artifact helpers: `has_checkpoint_artifact()`, `download_run_split()`
  (explicit, user-invoked recovery of a prior MLflow run's persisted split —
  see `dlkit.engine.tracking.tracking.md`)
- run-based checkpoint selection helpers: `find_latest_run_id()`,
  `find_child_run_ids()`, `download_checkpoint_artifact()` — thin wrappers
  over `dlkit.engine.tracking.run_queries`/`checkpoint_recovery` (see
  `dlkit.engine.tracking.tracking.md`); none is invoked automatically by
  `evaluate()`
- run-based checkpoint selection types: `MultiRunResult`, `RunCheckpoint`,
  `LatestRunCheckpoint` (re-exported from `dlkit.common`/
  `dlkit.common.checkpoint_source` for use with `evaluate()`'s
  `run_checkpoint` parameter — see `dlkit.interfaces.inference.inference.md`)

## Example
```python
from dlkit.interfaces.api import converge, execute, train
from dlkit.interfaces.api.domain import ConvergenceOverrides, ExecutionOverrides, TrainingOverrides

result = train(settings, overrides=TrainingOverrides(epochs=25, learning_rate=1e-3))
result = converge(settings, overrides=ConvergenceOverrides(sizes=(100, 500, 1000)))
result = execute(settings, overrides=ExecutionOverrides(trials=10))
```
