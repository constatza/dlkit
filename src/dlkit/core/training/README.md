# Training Guide: Losses With Extra Keyword Arguments

This guide explains how to configure losses and metrics that require additional
batch tensors beyond the default `(predictions, target)` pair.

## Default Contract

During wrapper execution, loss functions are called as:

```python
loss_fn(predictions, target, **extra_kwargs)
```

Where:
- `predictions` comes from the model output.
- `target` is selected from the batch (defaults to the first configured target).
- `extra_kwargs` are optional routed tensors from `features.*` or `targets.*`.

## Routing Sources

There are two ways to route extra keyword arguments to the loss function.

### 1. Automatic Routing With `DataEntry.loss_input`

Set `loss_input` on a feature/target entry to auto-route that tensor:

```python
from dlkit.tools.config.data_entries import Feature, Target

entries = (
    Feature(name="x", path="data/x.npy"),
    Feature(name="K", path="data/stiffness.npy", model_input=False, loss_input="matrix"),
    Target(name="y", path="data/y.npy"),
)
```

In this example, the wrapper calls:

```python
loss_fn(predictions, target, matrix=batch["features", "K"])
```

### 2. Explicit Routing With `extra_inputs`

Configure explicit routes in `WRAPPER.loss_function.extra_inputs`:

```toml
[WRAPPER]
loss_function = { name = "EnergyNormLoss", target_key = "targets.y", extra_inputs = [
  { key = "features.K", arg = "matrix" }
] }
```

## Target Selection

Loss target selection uses:

1. `WRAPPER.loss_function.target_key` if configured.
2. Otherwise, the first configured target entry.

Valid key format is always `features.<entry_name>` or `targets.<entry_name>`.

## Context Features (`model_input = false`)

For tensors needed by loss/metrics but not by the model forward pass, use
`model_input = false`.

```python
Feature(name="K", path="data/stiffness.npy", model_input=False, loss_input="matrix")
```

This keeps `K` in the batch, but excludes it from positional model invocation.

## Metric Routing

Metrics use the same routing primitives:

- `WRAPPER.metrics[*].target_key`
- `WRAPPER.metrics[*].extra_inputs`

Example:

```toml
[WRAPPER]
metrics = [
  { name = "EnergyNormError", target_key = "targets.y", extra_inputs = [
    { key = "features.K", arg = "matrix" }
  ] }
]
```

## Validation and Error Behavior

The wrapper validates routing at construction time and fails fast for invalid
configurations.

- Duplicate `loss_input` argument names across entries raise `ValueError`.
- Overlap between `DataEntry.loss_input` and `loss_function.extra_inputs` raises `ValueError`.
- Invalid key format (not `features.*` or `targets.*`) raises `ValueError`.
- Loss kwarg signature mismatch raises `ValueError` when introspection is possible.
  - If the loss callable accepts `**kwargs`, all arg names are allowed.
  - Uninspectable callables skip signature validation.

## End-to-End Example

```toml
[DATASET]
features = [
  { name = "x", path = "data/x.npy" },
  { name = "K", path = "data/stiffness.npy", model_input = false, loss_input = "matrix" },
]
targets = [
  { name = "y", path = "data/y.npy" },
]

[WRAPPER]
loss_function = { name = "EnergyNormLoss", target_key = "targets.y" }
metrics = [
  { name = "EnergyNormError", target_key = "targets.y", extra_inputs = [
    { key = "features.K", arg = "matrix" }
  ] }
]
```

Behavior:
- Model invocation uses only `x` as input.
- Loss receives `matrix=batch["features", "K"]` automatically from `loss_input`.
- Metric receives `matrix` via explicit `extra_inputs`.
