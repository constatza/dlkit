# API Functions Module

`dlkit.interfaces.api.functions` is a façade over runtime-owned entrypoints.

## Responsibilities
- expose the public workflow functions
- accept TypedDict-style override payloads
- coerce known string paths to `Path`
- forward normalized requests to runtime entrypoints

## Public Functions
- `train()`
- `optimize()`
- `execute()`
- `validate_config()`
- `generate_template()`
- `validate_template()`

## Example
```python
from dlkit.interfaces.api import execute, train

result = train(settings, overrides={"epochs": 25, "learning_rate": 1e-3})
result = execute(settings, overrides={"trials": 10})
```
