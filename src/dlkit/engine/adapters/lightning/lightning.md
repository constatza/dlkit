# Lightning Adapter Architecture

`dlkit.engine.adapters.lightning` owns PyTorch Lightning integration for DLKit.
That includes both wrapper classes and the Lightning-specific datamodules under
`datamodules/`.

Protocol-composed wrappers remain thin coordinators: all computation is
delegated to injected SOLID protocol objects.

---

## Architecture Overview

```
StandardLightningWrapper.__init__
  │
  ├─► _build_model_from_settings()    →  nn.Module
  │
  ├─► IModelInvoker ── TensorDictModelInvoker  (default positional dispatch via TensorDictModule)
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

For manual optimization programs, `training_step()` remains a thin coordinator:
- it builds a loss closure
- it delegates stepping to the optimization controller
- the controller uses the wrapper only through the narrow manual-host seam
  (`manual_backward()` and `optimizers(use_pl_optimizer=True)`)

**Files**

| File | Purpose |
|---|---|
| `protocols.py` | SOLID protocols (ISP-compliant interfaces) |
| `components.py` | Concrete protocol implementations + `WrapperComponents` value object (FR-2) |
| `base.py` | `ProcessingLightningWrapper` — pure Lightning coordinator |
| `datamodules/` | Lightning `DataModule` implementations for array, graph, and timeseries datasets |
| `callbacks.py` | Lifecycle callbacks such as transform fitting, epoch metric logging, explicit checkpoint-dir routing, and prediction writers |
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

Callback policy:
- callbacks never inspect ambient MLflow run state
- checkpoint routing receives an explicit local directory
- prediction writers only persist files to an explicit local directory and may
  record typed produced-artifact descriptors for later publication

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
from dlkit.infrastructure.config.data_entries import Feature, Target, ContextFeature

# Feature — fed to the model (model_input=True by default)
# Entry name must match the model.forward() parameter name.
Feature(name="x", path="data/features.npy")

# Multiple features — dispatched as kwargs: model(x=x_tensor, z=z_tensor)
Feature(name="x", path="data/features_x.npy")
Feature(name="z", path="data/features_z.npy")

# Context feature — in batch for loss/metric use, NOT passed to model
ContextFeature(name="A", path="data/stiffness.npy")  # model_input=False

# Target — extracted by loss/metric routing
Target(name="y", path="data/targets.npy")
```

`model_input` controls whether a feature is dispatched to `model.forward()`:

| `model_input` value | Dispatch style | Example |
|---|---|---|
| `True` (default) | kwarg, name == forward param | `model(x=x_tensor, z=z_tensor)` |
| `False` | excluded from model call | context feature for loss only |

Named features are dispatched as keyword arguments — `entry.name` is used directly
as the `forward()` parameter name. Config-list order does not affect routing.
`validate_forward_signature` checks the model signature at wrapper init time and
raises `ValueError` if a feature name has no matching parameter.

For NPZ inputs, the entry `name` is used as the array key.

---

## Single Input / Single Output

Minimal configuration for `model(x) → y`:

**TOML**:
```toml
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "data/features.npy"

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"
```

The wrapper automatically:
- Builds `NamedBatchTransformer(feature_chains={"x": Identity()}, target_chains={"y": Identity()})`
- Builds `TensorDictModelInvoker(in_keys=[("features","x")])` → calls `model(batch["features","x"])`
- Builds `RoutedLossComputer(loss_fn, target_key=None, default_target_key="y")` → `loss(preds, batch["targets","y"])`

---

## Multi-Input, Single Output

For `model(x, z) → y` (two feature arrays):

**TOML**:
```toml
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "data/features_x.npy"

[[DATASET.features]]
name = "z"
path = "data/features_z.npy"

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"
```

With both entries marked `model_input=true`, invocation dispatches by name:
`model(x=batch["features","x"], z=batch["features","z"])`.
Config-list order does not affect which tensor binds to which parameter.

For DeepONet-style branch/trunk inputs (model must declare `forward(u, query_coords)`):
```toml
[[DATASET.features]]
name = "u"
path = "..."
field_role = "feature"
model_input = true

[[DATASET.features]]
name = "query_coords"
path = "..."
field_role = "target_coordinates"
model_input = true
```
Invocation: `model(u=batch["features","u"], query_coords=batch["features","query_coords"])`.

---

## Loss and Metric Routing

Extra tensors (e.g. a stiffness matrix `A`) can be passed to loss/metric functions as
keyword arguments via `extra_inputs` / `target_key`:

**TOML**:
```toml
[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "data/features.npy"

[[DATASET.features]]
name = "A"
path = "data/stiffness.npy"
model_input = false

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"

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
from dlkit.engine.adapters.lightning.protocols import IModelInvoker
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
`StandardLightningWrapper.configure_callbacks()`. `NamedBatchTransformer.fit()`
handles three cases:

- Transforms implementing `IncrementalFittableTransform` (e.g. `IncrementalPCA`,
  `StandardScaler`, `MinMaxScaler`) are fitted batch-by-batch via
  `reset_fit_state → update_fit → finalize_fit`.
- Transforms implementing `_FittableFromDataloader` (i.e. `TransformChain`) use
  `fit_from_dataloader`, which materialises the full dataset for non-incremental
  algorithms like `PCA`, `ICA`, and `TruncatedSVD` in a single `fit(full_data)` call.
- Transforms that are already fitted are skipped.

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
    "feature_names": ["x"],          # model-input entry names (for inference transform lookup)
    "forward_arg_map": {"x": "x"},   # {kwarg_name: feature_name}; empty dict = positional dispatch
    "predict_target_key": "y",        # target whose inverse transform is applied at predict time
    "model_family": "dlkit_nn",
    "target_names": ["y"],
}
```

`dlkit_metadata` no longer carries a checkpoint `version` field. Loaders require
the metadata block itself and normalize older `model_settings` payloads into the
flat DTO shape above before reconstruction.

Serialized transform specs inside `entry_configs[*].transforms` are treated as
checkpoint metadata, not as already-instantiated settings objects. Transform
construction normalizes those mappings into typed `TransformSettings` before
applying runtime module defaults and creating transform modules.

`feature_names` is used by `CheckpointPredictor` to look up the correct feature
transform chain. `forward_arg_map` records the kwarg binding used during training —
`CheckpointPredictor.predict(x=tensor)` resolves `forward_arg_map["x"] = "x"` to
find the right transform and then calls `model(x=transformed_tensor)`. An empty
`forward_arg_map` indicates positional dispatch.

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
placeholders. Graph wrappers still share the same optimization-controller builder
as standard wrappers, so staged/concurrent optimizer policies resolve manual vs
automatic optimization through one source of truth.
