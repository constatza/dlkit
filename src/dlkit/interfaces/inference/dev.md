# Inference Module — Developer Reference

Technical architecture notes, design decisions, and implementation rationale for
`src/dlkit/interfaces/inference/`.

---

## Module Layout

```
interfaces/inference/
├── __init__.py         # Public API re-exports
├── api.py              # load_model(), validate_checkpoint(), get_checkpoint_info()
├── config.py           # PredictorConfig, ModelState dataclasses
├── predictor.py        # IPredictor protocol + CheckpointPredictor implementation
├── loading.py          # Checkpoint / model loading utilities
├── shapes.py           # Shape-spec inference from checkpoint / dataset
└── transforms.py       # Transform chain reconstruction and application helpers
```

Seven files. No hexagonal layers, no use-case objects, no adapters. All loading logic
is integrated directly into `CheckpointPredictor.load()`.

---

## Design: `predict(*args, **kwargs)`

### Why not `predict(dict)`?

The previous API was `predict(inputs: dict[str, Tensor] | Tensor)`.
Four problems:

1. **Mismatched mental model** — users think of `model.forward(x, edge_attr=ea)`, not
   `predictor.predict({"x": x, "edge_attr": ea})`. Wrapping tensors into a dict for
   a library call is non-idiomatic Python.

2. **Name coupling** — the dict key `"x"` must match the entry name in the training
   config. This couples inference callers to training configuration details they
   shouldn't need to know.

3. **No multi-output path** — returning a wrapped prediction object forced
   the caller to dig through a wrapper object to get a tensor. For VAE-style models
   returning `(recon, mu, logvar)`, there was no clean way to surface the extra
   outputs.

4. **Inconsistency with training** — the training path dispatches to `model.forward()`
   using kwargs via `TensorDictModelInvoker`. The inference path should mirror that
   dispatch exactly.

### New contract

```python
def predict(
    self,
    *args: torch.Tensor,
    **kwargs: torch.Tensor,
) -> torch.Tensor | TensorDict | tuple[Any, ...]:
```

- **Positional args** → passed as `model(arg0, arg1, ...)`.
  Arg `i` is transformed using the chain registered under `feature_names[i]`.
- **Keyword args** → passed as `model(key=tensor, ...)`.
  Kwarg `k` is transformed using the chain registered under `k`.
- **Returns** `Tensor` for single-output models; `tuple[Tensor, ...]` for
  multi-output; `TensorDict` when `forward()` itself returns a TensorDict.

The caller always receives raw tensors — no wrapper objects to unwrap.

### Dispatch mirrors training

`_classify_feature_entries()` in `components.py` is the single source of truth for
how entry configs are mapped to `model.forward()` parameters:

```
model_input=True          → kwarg, key = entry name     model(x=tensor)
model_input=int / "0"…   → positional, sorted by index  model(tensor, ...)
model_input="name"        → kwarg, key = "name"         model(name=tensor)
model_input=False/None    → excluded                    (not passed)
```

This function is used by both:
- `_build_invoker_from_entries()` — builds `TensorDictModelInvoker` for training
- `_ordered_model_input_names()` — computes the `feature_names` list stored in
  the checkpoint's `dlkit_metadata`

Because both use the same classifier, the inference caller can reproduce training
dispatch exactly by calling `predictor.predict(**feature_kwargs)` with kwargs derived
from the checkpoint's `feature_names`.

---

## ModelState: What Gets Stored

`ModelState` is a frozen, slot-backed keyword-only dataclass that holds all inference
state after `load()`:

| Field | Type | Purpose |
|---|---|---|
| `model` | `nn.Module` | Loaded model in eval mode on target device |
| `device` | `str` | "cpu", "cuda", "mps" |
| `shape_spec` | `ShapeSummary \| None` | Shapes used when building the model |
| `feature_transforms` | `dict[str, TransformChain] \| None` | Named feature chains |
| `target_transforms` | `dict[str, TransformChain] \| None` | Named target chains |
| `metadata` | `dict` | Raw `dlkit_metadata` from checkpoint |
| `feature_names` | `tuple[str, ...]` | Ordered model-input names (dispatch order) |
| `predict_target_key` | `str` | Entry name whose inverse chain is applied to output |

`feature_names` and `predict_target_key` are read from `dlkit_metadata` (written by
`ProcessingLightningWrapper.on_save_checkpoint()`). Old checkpoints that predate this
field gracefully degrade: `feature_names` becomes `()` (empty tuple), and the
transform helpers skip silently.

---

## Output Normalization: Reusing `_unpack_model_output`

`CheckpointPredictor.predict()` imports and calls `_unpack_model_output` from
`dlkit.core.models.wrappers.base`:

```python
from dlkit.core.models.wrappers.base import _unpack_model_output

raw_output = self._model_state.model(*args, **kwargs)
predictions_raw, latents_raw = _unpack_model_output(raw_output)
```

This handles every output shape that `forward()` can return:

| `forward()` return type | `predictions_raw` | `latents_raw` |
|---|---|---|
| `Tensor` | the tensor | `None` |
| `(Tensor,)` | `Tensor` | `None` |
| `(pred, lat)` | `pred` | `lat` |
| `(pred, lat0, lat1, ...)` | `pred` | `(lat0, lat1, ...)` |
| `TensorDict` with `"predictions"` key | `td["predictions"]` | `td.get("latents")` |
| Any other `Tensor` | same tensor | `None` |

After normalizing:
- Inverse target transform is applied to `predictions_raw` only.
- The final return is:
  - `predictions_raw` if `latents_raw is None` (single output)
  - `(predictions_raw, *latents_raw)` if `latents_raw` is a tuple
  - `(predictions_raw, latents_raw)` if `latents_raw` is a single tensor

Using `_unpack_model_output` instead of custom logic ensures identical behavior to
the `predict_step` path inside the Lightning wrappers.

---

## Transform Application

### Input transforms

```python
def _apply_input_transforms(
    self,
    args: tuple[torch.Tensor, ...],
    kwargs: dict[str, torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
```

- For positional arg `i`: look up `feature_names[i]` in `feature_transforms`. If the
  name is present, call `transform(tensor)`; otherwise pass through.
- For kwarg `k`: look up `k` directly in `feature_transforms`.

This means positional callers benefit from `feature_names` ordering, while kwarg
callers can use any name they want as long as it matches an entry name.

### Output inverse transform

```python
def _apply_output_inverse_transform(
    self,
    predictions: torch.Tensor | TensorDict,
) -> torch.Tensor | TensorDict:
```

Uses `predict_target_key` to select a single chain from `target_transforms`. Only
applies if:

1. `target_transforms` is non-empty
2. `predict_target_key` is a non-empty string
3. The key is present in `target_transforms`
4. The selected chain is an `InvertibleTransform` instance (isinstance check, not duck
   typing — same guard used in `NamedBatchTransformer.inverse_transform_predictions`)

This is more precise than the old `apply_inverse_transforms()` which raised
`ValueError` for single tensor + multiple target transforms (ambiguous). The new
approach always uses the explicitly-configured `predict_target_key`, so there is no
ambiguity.

`predict_target_key` is determined at training time in `StandardLightningWrapper.__init__()`:
- Default: first target entry name.
- Override: derived from `LossComponentSettings.target_key` (the loss target).

This means the inverse transform is applied to the same target that the loss was
computed on — the primary prediction output.

---

## Checkpoint Metadata Written by Wrappers

`ProcessingLightningWrapper.on_save_checkpoint()` (in `base.py`) writes these fields
into `checkpoint["dlkit_metadata"]`:

```python
dlkit_metadata["feature_names"] = list(meta.feature_names)   # from WrapperCheckpointMetadata
dlkit_metadata["predict_target_key"] = meta.predict_target_key
```

`WrapperCheckpointMetadata.feature_names` is populated by `_ordered_model_input_names()`
in `standard.py`:

```python
def _ordered_model_input_names(feature_entries: list[DataEntry]) -> tuple[str, ...]:
    positional, kwarg_map = _classify_feature_entries(feature_entries)
    positional.sort(key=lambda x: x[0])
    return tuple(name for _, name in positional) + tuple(kwarg_map.keys())
```

The **order** is: positionals first (ascending index), then kwargs (config-insertion
order). This is the exact same order `TensorDictModelInvoker` uses during training,
so callers that reconstruct kwargs by iterating `feature_names` always get the right
dispatch.

---

## Precision Inference

`CheckpointPredictor.load()` calls `PrecisionService.infer_precision_from_model(model)`
after loading the model. This inspects the first parameter's dtype:

- `float32` → `PrecisionStrategy.FULL_32`
- `float64` → `PrecisionStrategy.FULL_64`
- etc.

The inferred precision is stored as `self._inferred_precision`. On every `predict()` call:

```python
precision_to_use = self._config.precision or self._inferred_precision
if precision_to_use is not None:
    ctx = precision_override(precision_to_use)
```

This ensures the data loading path (if any data was loaded inside the context) uses
the same dtype as the model weights. Explicit `config.precision` overrides the
inferred value — useful for forcing mixed precision on a float32 checkpoint.

---

## CLI Predict Path

`interfaces/cli/commands/predict.py` iterates a `predict_dataloader()` and calls
`predictor.predict(**feature_kwargs)`:

```python
features_td = batch["features"]    # TensorDict from dataloader
if feature_names:
    feature_kwargs = {
        name: features_td[name]
        for name in feature_names
        if name in features_td.keys()
    }
else:
    # Old checkpoint with no feature_names — use all keys (insertion order)
    feature_kwargs = {k: features_td[k] for k in features_td.keys()}

output = predictor.predict(**feature_kwargs)
prediction = output[0] if isinstance(output, tuple) else output
all_predictions.append(prediction)
```

The fallback (empty `feature_names`) handles checkpoints trained before this feature
was added. In that case all feature keys are forwarded as kwargs, which works for most
single-input models.

---

## What Was Removed

| Removed | Reason |
|---|---|
| `_name_predictions()` | Tried to guess target name from entry_configs metadata; wrong schema, fragile. The correct approach is direct Tensor return. |
| Dict input support | Non-idiomatic Python for a function call; masked mismatches between key names and entry names. |
| `InferenceResult` wrapper for `predict()` | Forced callers to unwrap `.predictions` just to get a tensor. Direct tensor/tuple return is cleaner. |
| `apply_transforms()` / `apply_inverse_transforms()` imports in predictor | Replaced by `_apply_input_transforms` and `_apply_output_inverse_transform` which use `predict_target_key` directly instead of guessing. |

---

## Testing

| File | Coverage |
|---|---|
| `tests/interfaces/inference/test_simplified_predictor.py` | Load, precision inference, positional predict, kwarg predict, context manager, unload |
| `tests/interfaces/inference/test_checkpoint_utils.py` | `extract_state_dict`, dtype detection |
| `tests/integration/test_transforms_persistence_and_inference.py` | End-to-end: train → save → load → predict with transforms, inverse transforms, manual path |
| `tests/integration/test_basic_integration.py` | Smoke test for the full inference workflow |

Key scenarios in the integration tests:
- `test_predictor_returns_original_space_by_default` — verifies that inverse transform
  is applied when `apply_transforms=True`
- `test_predictor_accepts_pretransformed_inputs_when_disabled` — verifies that raw
  model output (no inverse) is returned when `apply_transforms=False`
- `test_manual_inverse_matches_default_path` — confirms that manually applying
  `apply_inverse_chain()` to `apply_transforms=False` output matches the default path

---

## Future Work / Known Limitations

- **Batching inside `predict()`** — the current implementation does a single forward
  pass regardless of input size. Large inputs may OOM. `PredictorConfig.batch_size`
  is stored but not used for internal chunking yet.
- **TensorDict output inverse transform** — `_apply_output_inverse_transform` applies
  `chain.apply()` uniformly to the whole TensorDict when `predictions` is a TensorDict.
  For multi-head TensorDicts with heterogeneous targets this may be incorrect; a
  per-key lookup would be safer.
- **NumPy array inputs** — the old API auto-converted NumPy arrays; the new API
  expects `torch.Tensor`. Callers must convert manually with `torch.from_numpy()`.
