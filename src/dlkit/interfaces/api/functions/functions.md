# API Functions Module

`dlkit.interfaces.api.functions` is a façade over runtime-owned entrypoints.

## Responsibilities
- expose the public workflow functions
- accept strict Pydantic override payloads (`TrainingOverrides`, `OptimizationOverrides`, `ExecutionOverrides`)
- coerce known string paths to `Path`
- forward normalized requests to runtime entrypoints

## Public Functions
- `train()`
- `optimize()`
- `execute()`
- `validate_config()`
- `generate_template()`
- `validate_template()`
- logged-model helpers: `search_logged_models()`, `build_logged_model_uri()`, `load_logged_model()`
- registry helpers: `register_logged_model()`, `search_registered_models()`, `list_model_versions()`, `get_model_version()`, `set_registered_model_alias()`, `set_registered_model_version_tag()`, `set_registered_model_version_tags()`, `build_registered_model_uri()`, `load_registered_model()`
- artifact helpers: `has_checkpoint_artifact()`

## Example
```python
from dlkit.interfaces.api import execute, train
from dlkit.interfaces.api.domain import TrainingOverrides, ExecutionOverrides

result = train(settings, overrides=TrainingOverrides(epochs=25, learning_rate=1e-3))
result = execute(settings, overrides=ExecutionOverrides(trials=10))
```
