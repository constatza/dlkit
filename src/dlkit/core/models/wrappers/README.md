# Lightning Wrappers Architecture

Protocol-composed PyTorch Lightning wrappers for DLKit models. Each wrapper is a
thin coordinator: all computation is delegated to injected SOLID protocol objects.

---

## Architecture Overview

```
StandardLightningWrapper.__init__
  │
  ├─► _build_model_from_settings()    →  nn.Module
  │
  ├─► IModelInvoker ── TensorDictModelInvoker  (kwarg/positional dispatch via TensorDictModule)
  ├─► ILossComputer ── RoutedLossComputer      (named key → loss fn kwargs)
  ├─► IMetricsUpdater─ RoutedMetricsUpdater    (per-metric routing, no MetricCollection.update)
  └─► IBatchTransformer NamedBatchTransformer  (named ModuleDict chains)
        │
        └─► IFittableBatchTransformer (fit() wired via configure_callbacks())

ProcessingLightningWrapper (base)
  _run_step():      abstract template hook implemented by each wrapper family
  training_step():  _run_step → log
  validation_step: _run_step → update_metrics → log
  predict_step():  capture raw targets → transform → invoke → inverse_transform → TensorDict
```

**Files**

| File | Purpose |
|---|---|
| `protocols.py` | SOLID protocols (ISP-compliant interfaces) |
| `components.py` | Concrete protocol implementations + `WrapperComponents` value object (FR-2) |
| `base.py` | `ProcessingLightningWrapper` — pure Lightning coordinator |
| `callbacks.py` | Lifecycle callbacks such as transform fitting |
| `checkpoint_dto.py` | Checkpoint metadata normalization helpers |
| `standard.py` | `StandardLightningWrapper` — tensor/TensorDict workflows |
| `graph.py` | `GraphLightningWrapper` — PyG Data/Batch workflows |
| `timeseries.py` | `TimeSeriesLightningWrapper` — tuple-batch workflows |
| `factories.py` | `WrapperFactory` — detects model family, returns correct wrapper |
| `security.py` | Checkpoint security — `configure_checkpoint_loading()`, `register_dlkit_safe_globals()` |

**Dependency Injection (FR-2)**

Core wrappers no longer call `FactoryProvider` directly. All factory calls are
centralised in `runtime/workflows/factories/component_builders.py`. `BuildFactory`
strategies pre-build a `WrapperComponents` value object and pass it to
`WrapperFactory.create_*()`, which forwards it to each wrapper's `__init__`.

```python
# runtime layer — component_builders.py
components = build_wrapper_components(settings, entry_configs)

# runtime layer — build_factory.py
wrapper = WrapperFactory.create_standard_wrapper(model, entry_configs, components, ...)

# core layer — standard.py (receives components, never calls FactoryProvider)
class StandardLightningWrapper(ProcessingLightningWrapper):
    def __init__(self, model, entry_configs, components: WrapperComponents, ...): ...
```

---

## TensorDict Batch Format

Every batch flowing through the standard wrapper is a nested `TensorDict`:

```python
TensorDict(
    {
        "features": TensorDict(
            {"x": tensor([B, F]), "A": tensor([B, N, N])},  # context feature
            batch_size=[B],
        ),
        "targets": TensorDict({"y": tensor([B, T])}, batch_size=[B]),
    },
    batch_size=[B],
)
```

Access patterns:
```python
batch["features", "x"]  # single tensor
batch["features"]["x"]  # equivalent
batch["targets", "y"]  # target tensor
batch.batch_size  # [B]
batch.to("cuda")  # recursive device transfer — no custom collation needed
```

The standard wrapper no longer accepts DLKit's legacy custom `Batch` transport.
For non-graph workflows, datasets and dataloaders must yield nested `TensorDict`
objects with top-level `"features"` and `"targets"` entries.

---

## DataEntry Configurations

```python
from dlkit.tools.config.data_entries import Feature, Target, ContextFeature

# Feature — fed to the model (model_input=True by default)
Feature(name="x", path="data/features.npy")

# Multiple features — default dispatch is kwargs: model(x=x_tensor, z=z_tensor)
Feature(name="x", path="data/features_x.npy")
Feature(name="z", path="data/features_z.npy")

# Context feature — in batch for loss/metric use, NOT passed to model
ContextFeature(name="A", path="data/stiffness.npy")  # model_input=False

# Target — extracted by loss/metric routing
Target(name="y", path="data/targets.npy")
```

`model_input` controls how (and whether) a feature is dispatched to `model.forward()`:

| `model_input` value | Dispatch style | Example |
|---|---|---|
| `True` (default) | kwarg, key = entry name | `model(x=tensor)` |
| `0`, `1`, … (int) | positional, sorted by index | `model(tensor0, tensor1)` |
| `"name"` (non-digit str) | kwarg, key = `"name"` | `model(name=tensor)` |
| `False` / `None` | excluded from model call | context feature for loss only |

`_classify_feature_entries()` in `components.py` encodes these rules and is shared by
the training invoker (`TensorDictModelInvoker`) and the inference `feature_names`
metadata so both paths use identical dispatch ordering.

For NPZ inputs, the entry `name` is used as the array key.

---

## Single Input / Single Output

Minimal configuration for `model(x) → y`:

**TOML**:
```toml
[DATASET]
features = [{ name = "x", path = "data/features.npy" }]
targets = [{ name = "y", path = "data/targets.npy" }]

[MODEL]
name = "LinearNet"
in_features = 32
out_features = 8

[WRAPPER]
loss_function = { name = "mse" }
```

The wrapper automatically:
- Builds `NamedBatchTransformer(feature_chains={"x": Identity()}, target_chains={"y": Identity()})`
- Builds `TensorDictModelInvoker(kwarg_in_keys={"x": ("features","x")})` → calls `model(x=batch["features","x"])`
- Builds `RoutedLossComputer(loss_fn, target_key=None, default_target_key="y")` → `loss(preds, batch["targets","y"])`

---

## Multi-Input, Single Output

For `model(x, z) → y` (two feature arrays):

**TOML**:
```toml
[DATASET]
features = [
  { name = "x", path = "data/features_x.npy" },
  { name = "z", path = "data/features_z.npy" },
]
targets = [{ name = "y", path = "data/targets.npy" }]

[WRAPPER]
loss_function = { name = "mse" }
```

Default (`model_input=True`) dispatches as kwargs: `model(x=batch["features","x"], z=batch["features","z"])`.

For explicit positional ordering, set `model_input` to an integer:
```toml
features = [
  { name = "x", path = "...", model_input = 0 },
  { name = "z", path = "...", model_input = 1 },
]
```
Invocation: `model(batch["features","x"], batch["features","z"])` — positionals sorted by index.

---

## Loss and Metric Routing

Extra tensors (e.g. a stiffness matrix `A`) can be passed to loss/metric functions as
keyword arguments via `extra_inputs` / `target_key`:

**TOML**:
```toml
[DATASET]
features = [
  { name = "x", path = "data/features.npy" },
  { name = "A", path = "data/stiffness.npy", model_input = false },
]
targets = [{ name = "y", path = "data/targets.npy" }]

[WRAPPER]
loss_function = { name = "EnergyNormLoss", target_key = "targets.y", extra_inputs = [
  { key = "features.A", arg = "matrix" }
] }
metrics = [
  { name = "EnergyNormError", target_key = "targets.y", extra_inputs = [
    { key = "features.A", arg = "matrix" }
  ] }
]
```

At each step:
```python
# RoutedLossComputer.compute():
loss(predictions, batch["targets", "y"], matrix=batch["features", "A"])

# RoutedMetricsUpdater.update():
metric.update(predictions, batch["targets", "y"], matrix=batch["features", "A"])
```

`key` format: `"features.<entry_name>"` or `"targets.<entry_name>"`.
`arg` is the keyword argument name on the loss/metric function.

---

## Custom Protocol Implementations

Replace any protocol without modifying the wrapper:

```python
from dlkit.core.models.wrappers.protocols import IModelInvoker
import torch.nn as nn
from tensordict import TensorDict
from typing import Any


class KwargModelInvoker:
    """Calls model with keyword arguments and writes output to batch."""

    def __init__(self, feature_keys: tuple[str, ...]) -> None:
        self._feature_keys = feature_keys

    def invoke(self, model: nn.Module, batch: TensorDict) -> TensorDict:
        kwargs = {k: batch["features", k] for k in self._feature_keys}
        predictions = model(**kwargs)
        batch["predictions"] = predictions
        return batch
```

Then pass it directly to `ProcessingLightningWrapper.__init__`:
```python
wrapper = ProcessingLightningWrapper(
    model=my_model,
    model_invoker=KwargModelInvoker(("x", "z")),
    loss_computer=my_loss_computer,
    metrics_updater=my_metrics_updater,
    batch_transformer=my_batch_transformer,
    optimizer_settings=optimizer_settings,
    predict_target_key="y",
)
```

---

## Transform Chains

Each data entry can have a list of transforms that are composed into a `TransformChain`:

```toml
[DATASET]
features = [
  { name = "x", path = "data/features.npy", transforms = [
    { name = "StandardScaler" },
    { name = "PCA", n_components = 16 },
  ] }
]
```

`NamedBatchTransformer` stores these as `nn.ModuleDict` entries under their entry name,
so state dict keys are stable:

```
_batch_transformer._feature_chains.x.transforms.0._fitted   # StandardScaler
_batch_transformer._feature_chains.x.transforms.1._fitted   # PCA
```

Fittable transforms are fitted automatically through
`StandardLightningWrapper.configure_callbacks()` using a streaming multi-pass
dataloader flow (no full `torch.cat` buffering):

- Incremental-capable transforms (currently `StandardScaler`, `MinMaxScaler`) are
  fitted batch-by-batch.
- Non-incremental fittable transforms must already be fitted; otherwise fitting
  fails fast with a clear error.
- Current policy: online fitting for unfitted `PCA` is rejected (`TODO: incremental PCA`).

---

## Checkpoint Format

DLKit checkpoints store a `dlkit_metadata` dict alongside the standard Lightning
`state_dict`:

```python
checkpoint["dlkit_metadata"] = {
    "wrapper_type": "StandardLightningWrapper",
    "model_settings": {
        "name": "LinearNet",
        "module_path": "...",
        "resolved_init_kwargs": {...},
        "all_hyperparams": {...},
    },
    "entry_configs": [{"name": "x", "class_name": "Feature", "transforms": [...]}, ...],
    "shape_summary": {"in_shapes": [[32]], "out_shapes": [[8]]},
    "feature_names": ["x"],  # model-input entries in dispatch order (for inference)
    "predict_target_key": "y",  # target whose inverse transform is applied at predict time
    "model_family": "dlkit_nn",
    "target_names": ["y"],
}
```

`dlkit_metadata` no longer carries a checkpoint `version` field. Loaders require
the metadata block itself and normalize older `model_settings` payloads into the
flat DTO shape above before reconstruction.

`feature_names` is used by `CheckpointPredictor` to map positional args in
`predictor.predict(tensor0, tensor1)` to the correct feature transform chain and
to reconstruct the correct `model.forward()` dispatch order. The list only includes
model-input entries (those with `model_input ≠ False/None`), in the same order as
`TensorDictModelInvoker` uses during training.

---

## Extension Points

| Concern | Protocol | Default impl | How to replace |
|---|---|---|---|
| Model invocation | `IModelInvoker` | `TensorDictModelInvoker` | Implement `invoke(model, batch) → TensorDict` |
| Loss computation | `ILossComputer` | `RoutedLossComputer` | Implement `compute(preds, batch)` |
| Metric tracking | `IMetricsUpdater` | `RoutedMetricsUpdater` | Implement `update/compute/reset` |
| Batch transforms | `IBatchTransformer` | `NamedBatchTransformer` | Implement `transform/inverse_transform_predictions` |
| Fittable transforms | `IFittableBatchTransformer` | `NamedBatchTransformer` | Add `fit/is_fitted` on top of above |

Graph models bypass these protocols entirely — `GraphLightningWrapper` overrides
all step methods and uses null sentinels (`_NullModelInvoker`, etc.) as base-class
placeholders.
