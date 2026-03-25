# Inference Module

Stateful checkpoint predictor — load once, predict many times.

No configuration files required. All model settings, shapes, and transform chains are
reconstructed automatically from the checkpoint.

Inference requires checkpoints with `dlkit_metadata`. Legacy checkpoints without
that metadata block are no longer supported.

Inference is now separate from the unified workflow executor: use `load_model()`
for checkpoint inference, and keep `train()`, `optimize()`, or `execute()` for
training-family workflows only.

---

## Quick Start

```python
from dlkit import load_model
import torch

# Load checkpoint (expensive — do this once)
predictor = load_model("model.ckpt", device="auto")

# Predict — mirrors model.forward() exactly
output = predictor.predict(x=torch.randn(32, 10))  # → Tensor (32, out_dim)

# Clean up
predictor.unload()
```

---

## Public API

| Symbol | Kind | Purpose |
|---|---|---|
| `load_model()` | function | Primary factory — creates a loaded `CheckpointPredictor` |
| `CheckpointPredictor` | class | Stateful predictor (load → predict × N → unload) |
| `IPredictor` | protocol | Interface for custom predictor implementations |
| `PredictorConfig` | frozen dataclass | Keyword-only configuration for `CheckpointPredictor` |
| `validate_checkpoint()` | function | Check checkpoint integrity without loading |
| `get_checkpoint_info()` | function | Extract metadata without loading the model |
| `PredictorNotLoadedError` | exception | Raised when `predict()` called before `load()` |

`CheckpointInfo` exposes metadata presence, `model_family`, and `wrapper_type`.
It no longer includes a checkpoint `version` field.

---

## `load_model()` — Factory Function

```python
predictor = load_model(
    checkpoint_path="path/to/model.ckpt",
    device="auto",  # "auto" | "cpu" | "cuda" | "mps"
    batch_size=32,  # stored in config; not used for internal chunking yet
    apply_transforms=True,  # apply feature/target transforms from checkpoint
    auto_load=True,  # load immediately (set False to defer)
    precision=None,  # override inferred precision (PrecisionStrategy | None)
)
```

`"auto"` device picks CUDA > MPS > CPU in that order.

---

## `predict(*args, **kwargs)` — Forward API

`predict()` mirrors `model.forward()` exactly. Use the same argument style you would
use to call the underlying PyTorch model:

```python
# Single positional arg (simplest case)
output = predictor.predict(torch.randn(32, 10))

# Single kwarg (most explicit)
output = predictor.predict(x=torch.randn(32, 10))

# Multi-input positional
output = predictor.predict(torch.randn(32, 10), torch.randn(32, 5))

# Multi-input kwargs (mixed is fine too)
output = predictor.predict(x=torch.randn(32, 10), edge_attr=torch.randn(32, 5))
```

### Return types

| Model output | `predict()` returns |
|---|---|
| Single `Tensor` | `Tensor` |
| `(pred, latent)` tuple | `(Tensor, Tensor)` |
| `(pred, lat0, lat1, ...)` | `(Tensor, Tensor, Tensor, ...)` |
| `TensorDict` with `"predictions"` | the `TensorDict` |

```python
# Single output
output = predictor.predict(x=x_tensor)  # Tensor

# VAE — recon + latent parameters
recon, mu, logvar = predictor.predict(x=x_tensor)

# Multi-output with latents
pred, *latents = predictor.predict(x=x_tensor)
```

---

## Transforms

When `apply_transforms=True` (default):

1. **Feature transforms** — applied to each input tensor before the forward pass.
   The transform for positional arg `i` is looked up by `feature_names[i]`; the
   transform for kwarg `k` is looked up by `k`.
2. **Model forward** — `model(*transformed_args, **transformed_kwargs)`
3. **Inverse target transform** — applied to the first output (primary prediction)
   to convert it back to original units.

```python
# Raw features → transformed → model → inverse-transformed → original space
output = predictor.predict(x=raw_features)

# Pre-normalized inputs → skip forward transform → raw model output
predictor_no_tf = load_model("model.ckpt", apply_transforms=False)
output = predictor_no_tf.predict(x=normalized_features)
```

Transform chains are reconstructed from the checkpoint's `state_dict`, including
fitted `MinMaxScaler`/`StandardScaler` statistics and any pre-fitted `PCA`
components saved at training time. Inference does not fit missing transform state.

Model reconstruction uses normalized checkpoint metadata and strict weight loading:
only `model.*` weights are loaded into the rebuilt model, and key mismatches fail
fast instead of being ignored.

---

## Context Manager

```python
with load_model("model.ckpt", device="cuda") as predictor:
    for batch in my_data:
        output = predictor.predict(x=batch)
        process(output)
# Model moved off GPU and memory freed automatically
```

---

## Direct Model Access

For workflows that need full PyTorch control:

```python
predictor = load_model("model.ckpt")
model = predictor.model  # standard nn.Module in eval mode

with torch.no_grad():
    output = model(x_tensor)

# Fine-tuning
for name, param in model.named_parameters():
    print(name, param.shape)
```

---

## Precision

Precision is inferred from the model parameter dtype (`float32`, `float64`, etc.) and
applied automatically via a thread-local context during `predict()`. Override:

```python
from dlkit.tools.config.precision.strategy import PrecisionStrategy

predictor = load_model("model.ckpt", precision=PrecisionStrategy.FULL_64)
```

---

## Checkpoint Validation

```python
from dlkit import validate_checkpoint, get_checkpoint_info

# Validate without loading
info = validate_checkpoint("model.ckpt")
assert info.valid_format and info.has_state_dict

# Inspect metadata
meta = get_checkpoint_info("model.ckpt")
print(meta.model_family, meta.wrapper_type)
```

---

## Developer Notes

See [`dev.md`](dev.md) for architectural decisions, dispatch semantics, output
normalization, and known limitations.
