# SOLID Architecture Redesign: Wrappers, Batch Container, Loss/Metric Routing

## Context

The current architecture has accumulated several SOLID violations and design gaps since the positional `Batch` migration:

1. **God-class wrappers** – `ProcessingLightningWrapper` handles model building, precision, metrics, loss, optimizer, scheduler, checkpoint persistence, and logging in one class.
2. **Loss/metric routing blocked** – `_compute_loss` and `_update_metrics` are hardwired to `(predictions, targets[0])`. There is no mechanism to pass extra batch data (e.g. the `matrix` arg of `energy_norm_loss` or `EnergyNormError.update`).
3. **Positional fragility** – `Batch(features=(t0,t1), targets=(t0,))` is anonymous; reasoning about which tensor is which requires counting config-insertion positions.
4. **`BareWrapper` still dict-based** – violates the typed `Batch` contract.
5. **`required_in_loss: bool`** in `DataEntry` is an unfinished stub that cannot route tensors.
6. **No "context feature" concept** – a feature loaded per sample for the loss/metric (e.g. stiffness matrix `A`) but not fed to the model cannot be represented; `_invoke_model` would blindly pass it to `model(x, A)`.
7. **`PositionalBatchTransformer` aligns by position** – fragile positional ModuleList breaks if entries are reordered.

---

## Key Design Decisions

### Decision 1: Replace `Batch` with `TensorDict` (breaking change, justified)

TensorDict replaces the anonymous `Batch(features=tuple, targets=tuple)`:
- **Named access** solves loss/metric routing: `batch["features", "A"]`
- **Recursive device transfer**: `batch.to(device)` moves all nested tensors → removes custom `transfer_batch_to_device` override
- **Lazy collation**: `lazy_stack(samples, dim=0)` replaces the manual `_collate_batch` loop
- **Nested structure**: `batch["features"]` → sub-TensorDict of all feature tensors
- **Autograd flows** through TensorDict operations
- **Official PyTorch ecosystem** dependency (safe, maintained)

**Nested key convention** (nested TensorDicts, not flat dot keys):
```python
TensorDict({
    "features": TensorDict({"x": t_x, "A": t_A}, batch_size=[]),
    "targets":  TensorDict({"y": t_y},             batch_size=[]),
}, batch_size=[])
```
Access: `batch["features", "x"]`  or  `batch["features"]["x"]`

### Decision 2: Model invocation is ALWAYS positional

Models receive features **positionally in config-insertion order**. Entry names are entirely independent of the model's `forward()` parameter names. The user controls invocation order by the order of Feature entries in config. This preserves all existing model behaviour and avoids kwarg name coupling.

```python
# Entry order: [Feature("x"), Feature("z")]
model(batch["features","x"], batch["features","z"])  # positional, always
```

Only `model_input=True` features are passed. Context features (`model_input=False`) are in the TensorDict for loss/metric use but never passed to the model.

### Decision 3: Loss/metric extra inputs are ALWAYS keyword args

Extra inputs to loss/metric functions are always keyword arguments. The first two positional slots (`predictions`, `target`) are always handled via `predictions_key`/`target_key`. This means every loss function must have named parameters (which all our custom functions already do).

```python
energy_norm_loss(predictions, target, matrix=batch["features","A"])
```

No need for integer-indexed positional `extra_inputs`.

### Decision 4: Named transform chains (breaking checkpoint format)

`PositionalBatchTransformer` replaces `_feature_chains: ModuleList` (positional) with `_feature_chains: nn.ModuleDict` (named by entry name). This eliminates the fragile position-alignment requirement.

Old state dict key: `_feature_chains.0.transforms.0._fitted`
New state dict key: `_batch_transformer._feature_chains.x.transforms.0._fitted`

This is a breaking checkpoint format change. The inference module's `load_transforms_from_checkpoint` gains a new format handler for the new keys.

### Decision 5: Inference module public API unchanged

`CheckpointPredictor.predict(inputs: dict | Tensor)` keeps its existing public signature. Internally it converts to TensorDict, uses the stored feature key ordering from checkpoint metadata, and applies transforms via named dict. The inference module does NOT use the Lightning wrapper's `predict_step` — it calls `model(...)` directly — so TensorDict adoption there is purely internal.

---

## Implementation Plan

### Step 1 – Add TensorDict dependency

- Add `tensordict` to `pyproject.toml` dependencies
- No conflicts with PyTorch ecosystem

---

### Step 2 – `DataEntry` evolution

**File**: `src/dlkit/tools/config/data_entries.py`

Add to `DataEntry` base class:
```python
model_input: bool = Field(default=True,
    description=(
        "If False, tensor is loaded into the batch and available for loss/metric, "
        "but is NOT passed as an argument to the model forward(). "
        "Use for context tensors (e.g. stiffness matrix for energy norm loss)."
    ))
```

Remove `required_in_loss: bool` — replaced by explicit routing in loss/metric settings.

A convenience factory: `ContextFeature(name, path, ...)` = `PathFeature(..., model_input=False)`.

---

### Step 3 – Loss/Metric input routing in config

**File**: `src/dlkit/tools/config/components/model_components.py`

```python
class LossInputRef(BasicSettings, frozen=True):
    """Maps a loss function kwarg to a TensorDict key in the batch."""
    arg: str   # kwarg name in the loss fn, e.g. "matrix"
    key: str   # batch key in "namespace.entry_name" format, e.g. "features.A"

class LossComponentSettings(ComponentSettings[Callable], frozen=True):
    name: str | Callable = "mse"
    module_path: str = "dlkit.core.training.functional"
    target_key: str | None = None               # None → first targets/ entry in config
    extra_inputs: tuple[LossInputRef, ...] = ()  # extra kwargs passed to loss fn

class MetricInputRef(BasicSettings, frozen=True):
    arg: str
    key: str

class MetricComponentSettings(ComponentSettings[Metric], frozen=True):
    name: str = "MeanSquaredError"
    module_path: str = "torchmetrics.regression"
    target_key: str | None = None
    extra_inputs: tuple[MetricInputRef, ...] = ()
```

**TOML example (energy norm loss with stiffness matrix from features):**
```toml
[TRAINING.loss_function]
name = "energy_norm_loss"
target_key = "targets.displacement"

[[TRAINING.loss_function.extra_inputs]]
arg = "matrix"
key = "features.stiffness_matrix"
```

---

### Step 4 – SOLID protocols

**New file**: `src/dlkit/core/models/wrappers/protocols.py`

```python
from typing import Any, Protocol, runtime_checkable
from torch import Tensor
import torch.nn as nn

@runtime_checkable
class ILossComputer(Protocol):
    """Single responsibility: compute scalar loss from predictions + named batch."""
    def compute(self, predictions: Tensor, batch: "TensorDict") -> Tensor: ...

@runtime_checkable
class IMetricsUpdater(Protocol):
    """Single responsibility: accumulate and expose metric state."""
    def update(self, predictions: Tensor, batch: "TensorDict", stage: str) -> None: ...
    def compute(self, stage: str) -> dict[str, Any]: ...
    def reset(self, stage: str) -> None: ...

@runtime_checkable
class IModelInvoker(Protocol):
    """Single responsibility: extract tensors from TensorDict and call model positionally."""
    def invoke(self, model: nn.Module, batch: "TensorDict") -> Tensor: ...

@runtime_checkable
class IBatchTransformer(Protocol):
    """Single responsibility: apply/invert transforms on TensorDict entries."""
    def transform(self, batch: "TensorDict") -> "TensorDict": ...
    def inverse_transform_predictions(self, predictions: Tensor, target_key: str) -> Tensor: ...
    def fit(self, dataloader: Any) -> None: ...
    def is_fitted(self) -> bool: ...
```

---

### Step 5 – Protocol implementations

**New file**: `src/dlkit/core/models/wrappers/components.py`

#### `StandardModelInvoker`
```python
class StandardModelInvoker:
    """Extracts model-input features from TensorDict in config-insertion order."""
    def __init__(self, feature_keys: tuple[str, ...]):
        # keys of model_input=True features in config order (determines positional order)
        self._feature_keys = feature_keys

    def invoke(self, model: nn.Module, batch: TensorDict) -> Tensor:
        tensors = tuple(batch["features", k] for k in self._feature_keys)
        match len(tensors):
            case 0: raise ValueError("No model-input features in batch")
            case 1: return model(tensors[0])
            case _: return model(*tensors)
```

#### `RoutedLossComputer`
```python
class RoutedLossComputer:
    """Routes batch keys to loss function kwargs per LossComponentSettings."""
    def __init__(
        self,
        loss_fn: Callable,
        spec: LossComponentSettings,
        default_target_key: str,      # first target entry name
    ):
        self._loss_fn = loss_fn
        # parse "namespace.name" format
        self._target_ns, self._target_name = _parse_key(spec.target_key or f"targets.{default_target_key}")
        self._extra = spec.extra_inputs

    def compute(self, predictions: Tensor, batch: TensorDict) -> Tensor:
        target = batch[self._target_ns, self._target_name].to(dtype=predictions.dtype)
        extra_kwargs = {
            ref.arg: batch[_parse_key(ref.key)]
            for ref in self._extra
        }
        return self._loss_fn(predictions, target, **extra_kwargs)
```

#### `RoutedMetricsUpdater`
```python
class RoutedMetricsUpdater:
    """Wraps MetricCollections and routes batch keys per MetricComponentSettings."""
    def __init__(
        self,
        val_metrics: MetricCollection,
        test_metrics: MetricCollection,
        specs: tuple[MetricComponentSettings, ...],
        default_target_key: str,
    ): ...

    def update(self, predictions: Tensor, batch: TensorDict, stage: str) -> None:
        mc = self._metrics[stage]
        # Each metric may have its own target_key and extra_inputs
        # Call mc.update(predictions, target, **extra_kwargs)
        ...
```

#### `NamedBatchTransformer` (replaces positional ModuleList pattern)
```python
class NamedBatchTransformer(nn.Module):
    """Applies named transform chains per entry key. dict[name → chain]."""

    def __init__(
        self,
        feature_chains: dict[str, nn.Module],   # {entry_name: chain}
        target_chains: dict[str, nn.Module],
    ):
        super().__init__()
        self._feature_chains = nn.ModuleDict(feature_chains)
        self._target_chains  = nn.ModuleDict(target_chains)

    def transform(self, batch: TensorDict) -> TensorDict:
        # Apply feature chains
        new_features = {
            k: self._feature_chains[k](batch["features", k])
               if k in self._feature_chains else batch["features", k]
            for k in batch["features"].keys()
        }
        new_targets = {
            k: self._target_chains[k](batch["targets", k])
               if k in self._target_chains else batch["targets", k]
            for k in batch["targets"].keys()
        }
        return TensorDict({
            "features": TensorDict(new_features, batch_size=batch.batch_size),
            "targets":  TensorDict(new_targets, batch_size=batch.batch_size),
        }, batch_size=batch.batch_size)

    def inverse_transform_predictions(self, predictions: Tensor, target_key: str) -> Tensor:
        chain = self._target_chains.get(target_key)
        if chain is not None and isinstance(chain, InvertibleTransform):
            return chain.inverse_transform(predictions)
        return predictions

    def fit(self, dataloader: Any) -> None:
        # Iterate loader, accumulate buffers by key, fit FittableTransforms
        ...

    def is_fitted(self) -> bool:
        for chain in [*self._feature_chains.values(), *self._target_chains.values()]:
            if isinstance(chain, FittableTransform) and not chain.fitted:
                return False
        return True
```

State dict keys become: `_batch_transformer._feature_chains.x.*` (named, stable, no position magic).

---

### Step 6 – Thin `ProcessingLightningWrapper`

**File**: `src/dlkit/core/models/wrappers/base.py` (rewrite)

```python
class ProcessingLightningWrapper(LightningModule, ABC):
    """Pure Lightning coordinator. All computation delegated to injected protocols."""

    def __init__(
        self, *,
        model: nn.Module,
        model_invoker: IModelInvoker,
        loss_computer: ILossComputer,
        metrics_updater: IMetricsUpdater,
        batch_transformer: IBatchTransformer,  # registered as nn.Module submodule
        optimizer_settings: OptimizerSettings,
        scheduler_settings: SchedulerSettings | None = None,
        # For checkpoint metadata only:
        model_settings: ModelComponentSettings,
        wrapper_settings: WrapperComponentSettings,
        entry_configs: tuple[DataEntry, ...] = (),
        shape_summary: ShapeSummary | None = None,
    ):
        super().__init__()
        self.model = model
        self._model_invoker = model_invoker
        self._loss_computer = loss_computer
        self._metrics_updater = metrics_updater
        self._batch_transformer = batch_transformer  # nn.Module → Lightning saves state
        ...

    def training_step(self, batch: TensorDict, batch_idx: int):
        batch = self._batch_transformer.transform(batch)
        predictions = self._model_invoker.invoke(self.model, batch)
        loss = self._loss_computer.compute(predictions, batch)
        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: TensorDict, batch_idx: int):
        batch = self._batch_transformer.transform(batch)
        predictions = self._model_invoker.invoke(self.model, batch)
        val_loss = self._loss_computer.compute(predictions, batch)
        self._metrics_updater.update(predictions, batch, stage="val")
        metrics = self._metrics_updater.compute("val")
        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    # test_step: same pattern
    # predict_step: transform → invoke → inverse_transform_predictions

    # NO model building, NO precision setup, NO MetricCollection creation
    # Retained: configure_optimizers, on_fit_start, on_save_checkpoint, on_load_checkpoint, LR sync
```

`transfer_batch_to_device` override is **deleted** — `TensorDict.to(device)` handles this.

---

### Step 7 – `StandardLightningWrapper` simplified

**File**: `src/dlkit/core/models/wrappers/standard.py`

- Becomes a concrete subclass that implements `forward()` and delegates to base
- `BareWrapper` is **removed** (dict-based, violates contract, no users in main flow)
- `on_fit_start` delegates to `self._batch_transformer.fit(loader)` (one-liner)

---

### Step 8 – `FlexibleDataset` returns `TensorDict`

**File**: `src/dlkit/core/datasets/flexible.py`

```python
def __init__(self, *, features, targets):
    ...
    self._feature_names: tuple[str, ...] = tuple(...)
    self._target_names:  tuple[str, ...] = tuple(...)
    self._feature_tensors: tuple[Tensor, ...] = ...   # same as before
    self._target_tensors:  tuple[Tensor, ...] = ...

def __getitem__(self, idx: int) -> TensorDict:
    from tensordict import TensorDict
    return TensorDict({
        "features": TensorDict(
            {name: (t if t.dim() == 0 else t[idx])
             for name, t in zip(self._feature_names, self._feature_tensors)},
            batch_size=[]
        ),
        "targets": TensorDict(
            {name: (t if t.dim() == 0 else t[idx])
             for name, t in zip(self._target_names, self._target_tensors)},
            batch_size=[]
        ),
    }, batch_size=[])
```

**DataLoader collation** – remove `_collate_batch` and `default_collate_fn_map` registration; add:
```python
def collate_tensordict(batch: list[TensorDict]) -> TensorDict:
    from tensordict import lazy_stack
    return lazy_stack(batch, dim=0)
```
Passed as `collate_fn=collate_tensordict` in `InMemoryModule` DataLoader creation.

---

### Step 9 – `BuildFactory` wires protocols

**File**: `src/dlkit/runtime/workflows/factories/build_factory.py`

```python
model_input_keys = tuple(e.name for e in entry_configs if is_feature_entry(e) and e.model_input)
all_feature_keys = tuple(e.name for e in entry_configs if is_feature_entry(e))
all_target_keys  = tuple(e.name for e in entry_configs if is_target_entry(e))

# Build transform chains (named dict, not positional list)
feature_chains = {e.name: _make_chain(e) for e in entry_configs if is_feature_entry(e)}
target_chains  = {e.name: _make_chain(e) for e in entry_configs if is_target_entry(e)}

batch_transformer = NamedBatchTransformer(feature_chains, target_chains)
model_invoker     = StandardModelInvoker(model_input_keys)

loss_fn      = FactoryProvider.create_component(wrapper_settings.loss_function, ...)
loss_computer = RoutedLossComputer(loss_fn, wrapper_settings.loss_function,
                                   default_target_key=all_target_keys[0])

val_metrics  = MetricCollection([...])
test_metrics = MetricCollection([...])
metrics_updater = RoutedMetricsUpdater(val_metrics, test_metrics,
                                       wrapper_settings.metrics, all_target_keys[0])

wrapper = StandardLightningWrapper(
    model=model,
    model_invoker=model_invoker,
    loss_computer=loss_computer,
    metrics_updater=metrics_updater,
    batch_transformer=batch_transformer,
    ...
)
```

---

### Step 10 – Inference module: internal TensorDict, public API unchanged

**File**: `src/dlkit/interfaces/inference/predictor.py`

Public API `predict(inputs: dict[str, Tensor] | Tensor)` is **unchanged**.

Internal changes:
1. Load feature key ordering from checkpoint metadata: `dlkit_metadata["feature_names"]`
2. Convert user's `dict/Tensor` → build TensorDict internally (not exposed to user)
3. Apply transforms via existing named-dict `apply_transforms` (already correct format)
4. Call `model(*[features_in_order])` positionally (same as now)

**File**: `src/dlkit/interfaces/inference/transforms.py`

`load_transforms_from_checkpoint` gains a new format branch:
```python
# NEW: named ModuleDict format (_batch_transformer._feature_chains.{name}.*)
has_named = any(k.startswith("_batch_transformer._feature_chains.") for k in state_dict)
```
Existing fallbacks for `_feature_chains.{idx}.*` (current positional), `fitted_feature_transforms.*`, `fitted_transforms.*` are retained for older checkpoints.

---

### Step 11 – Graph wrapper unchanged in spirit

`GraphLightningWrapper` bypasses TensorDict entirely — PyG `Data`/`Batch` objects flow directly to PyG operations. It overrides `training_step` etc. with its own decomposition. No change to graph batch handling.

---

### Step 12 – Metrics wrappers unchanged

`EnergyNormError.update(preds, target, matrix)` already has the right signature. The `RoutedMetricsUpdater` extracts the matrix from `batch["features","A"]` and passes it as `matrix=...` kwarg. **No change needed in the metric classes themselves.**

---

## Summary: Files to Create

| File | Purpose |
|------|---------|
| `src/dlkit/core/models/wrappers/protocols.py` | `ILossComputer`, `IMetricsUpdater`, `IModelInvoker`, `IBatchTransformer` |
| `src/dlkit/core/models/wrappers/components.py` | `StandardModelInvoker`, `RoutedLossComputer`, `RoutedMetricsUpdater`, `NamedBatchTransformer` |

---

## Summary: Files to Modify

| File | Change |
|------|--------|
| `pyproject.toml` | Add `tensordict` dependency |
| `src/dlkit/tools/config/data_entries.py` | Add `model_input: bool = True`; add `ContextFeature` alias; remove `required_in_loss` |
| `src/dlkit/tools/config/components/model_components.py` | Add `LossInputRef`, `MetricInputRef`; add `extra_inputs`/`target_key` to settings |
| `src/dlkit/core/datatypes/batch.py` | Keep for backward compat but deprecate; remove collation registration |
| `src/dlkit/core/datasets/flexible.py` | Store `_feature_names`/`_target_names`; return `TensorDict` from `__getitem__` |
| `src/dlkit/core/datamodules/array.py` | Pass `collate_fn=collate_tensordict` to DataLoader |
| `src/dlkit/core/models/wrappers/base.py` | Full rewrite as thin coordinator with injected protocols |
| `src/dlkit/core/models/wrappers/standard.py` | Simplify; remove `BareWrapper` |
| `src/dlkit/runtime/workflows/factories/build_factory.py` | Wire protocol implementations |
| `src/dlkit/interfaces/inference/predictor.py` | Internal TensorDict; public API unchanged |
| `src/dlkit/interfaces/inference/transforms.py` | Add named-ModuleDict format handler |

---

## Invariants Preserved

- `nn.Module` models remain pure `nn.Module` — no `LightningModule` inheritance required
- Individual transforms (`StandardScaler`, `PCA`, etc.) take `Tensor → Tensor` — unchanged
- `TransformChain` unchanged
- Graph wrappers bypass TensorDict entirely — PyG objects flow unchanged
- Metric functional implementations unchanged
- Config TOML format backward compatible (new fields have defaults)
- Inference module public API unchanged
- Checkpoint metadata structure preserved (v2.0 format) — **only state dict key paths change** for transform chains

---

## Verification

1. `pytest tests/ -x` → all 1896 tests pass
2. New test: `energy_norm_loss` via `LossInputRef(arg="matrix", key="features.A")` routing
3. New test: `EnergyNormError` metric via `MetricInputRef` routing
4. New test: `ContextFeature(model_input=False)` excluded from model invocation
5. New test: `NamedBatchTransformer` checkpoint save/load round-trip with named keys
6. New test: `predict(inputs=dict(...))` inference with named transform loading
7. Verify `batch.to(device)` works without `transfer_batch_to_device` override
