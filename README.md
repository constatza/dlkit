# DLKit

DLKit provides a thin, typed workflow layer on top of PyTorch and Lightning with:
- Flattened, typed configuration models (Pydantic) and a single TOML loader
- Strategy-based training (vanilla, MLflow) and hyperparameter optimization (Optuna)
- Simple dataset/datamodule factories with TensorDict-based shape inference
- A small CLI to create/validate configs and run workflows

## Installation

**DLkit** requires Python 3.12+. 
You can install the package using either `uv` or `pip`.

### Using `uv`

Ensure that `uv` is installed on your system. For official installation instructions tailored to your platform, please refer to the [uv documentation](https://docs.astral.sh/uv).

~~~bash
uv add git+https://github.com/constatza/dlkit
~~~

## Supported Data Formats

DLKit's `FlexibleDataset` supports multiple array file formats out of the box:

- **NumPy**: `.npy` (single array), `.npz` (multi-array archive)
- **PyTorch**: `.pt`, `.pth` (tensor files)
- **Text**: `.txt`, `.csv` (array data)
- **Sparse packs**: directory-based COO payload packs (defaults: `indices.npy`, `values.npy`, `nnz_ptr.npy`, `values_scale.npy`; names are configurable via `PackFiles` / `SparseFeature.files`)

### NPZ Multi-Array Support

For `.npz` files containing multiple arrays, the entry `name` is automatically used as the array key:

```toml
# data.npz contains arrays: "features", "targets", "latent"

[[DATASET.features]]
name = "features"  # Used as array key in NPZ
path = "data.npz"

[[DATASET.features]]
name = "latent"    # Loads different array from same file
path = "data.npz"

[[DATASET.targets]]
name = "targets"   # Used as array key
path = "data.npz"
```

See `src/dlkit/core/datasets/README.md` for detailed documentation and examples.

### Sparse Pack Context Features

For per-sample sparse matrices (for example, `A` matrices for energy-norm losses),
point a feature entry to a sparse pack directory. `FlexibleDataset` auto-detects
pack directories when `path` points to a folder containing sparse payload files.

```toml
[[DATASET.features]]
name = "matrix"
path = "data/matrix_pack"   # directory, not a .npy file
model_input = false
loss_input = "matrix"
```

Sparse pack API:
- `save_sparse_pack(...)`
- `open_sparse_pack(...)`
- `validate_sparse_pack(...)`

Scale contract:
- `A_original = A_stored * value_scale`
- `value_scale` defaults to `1.0`
- scaling is applied only when sparse readers use `denormalize=True` (including `SparseFeature(denormalize=True)`)

## Quick Start (CLI)

```bash
# Train a model
uv run dlkit train config.toml

# Run predictions on a dataset
uv run dlkit predict config.toml model.ckpt

# Hyperparameter optimization
uv run dlkit optimize config.toml --trials 50
```

## Example Configuration (auto-generated)

The project ships with auto-generated, commented templates (training, inference, mlflow, optuna). These are the single source of truth used by the CLI and synced into the repo.

- View the canonical training template in the repo: `examples/example_config.toml`
- Generate your own via CLI: `uv run dlkit config create --output config.toml --type training`
- Sync templates and generated examples: `uv run dlkit config sync-templates --write`

Training template (excerpt):

```toml
[SESSION]
# Human-readable run/session name (for logs and tracking)
name = "my_training_session"
# Inference mode flag (false = training)
inference = false
# Random seed for reproducibility
seed = 42
# Computation precision preset (use semantic aliases such as 'float32', 'mixed16', 'bf16')
precision = "float32"

[MODEL]
# Model class path or registry alias
name = "your.model.class"

[TRAINING]

[TRAINING.trainer]
# Maximum number of epochs (Lightning Trainer)
max_epochs = 100
# Hardware accelerator: cpu | gpu | auto | tpu
accelerator = "auto"

[DATAMODULE]
# DataModule class path or alias (dataflow loading)
name = "your.datamodule.class"

[DATASET]
# Dataset class path or alias
name = "your.dataset.class"

[MLFLOW]
# Flat, client-only tracking settings
experiment_name = "my-experiment"
run_name = "baseline"
tags = { team = "research" }
register_model = false

[EXTRAS]
# Optional user-defined helpers (ignored by core)
example_key = "value"
```

## Training and Optimization

- **Training**: Use `dlkit train` command (vanilla or with MLflow tracking)
- **Optimization**: Use `dlkit optimize` command for hyperparameter tuning with Optuna
- **MLflow tracking**: Add an `[MLFLOW]` section to your config; its presence enables tracking
- **Unified Python execution**: `execute()` routes only training and optimization workflows; use `load_model()` for inference

Examples:
- CLI (vanilla training): `uv run dlkit train config.toml --epochs 20`
- CLI (training with MLflow): `uv run dlkit train config_with_mlflow.toml --epochs 20`
- CLI (optuna optimization): `uv run dlkit optimize config.toml --trials 50`
- CLI (optuna with MLflow): `uv run dlkit optimize config_with_mlflow.toml --trials 50`
- Python (mlflow):
  ```python
  from dlkit.interfaces.api import train
  from dlkit.tools.io import load_settings
  cfg = load_settings("config_with_mlflow.toml")
  res = train(cfg, epochs=20, batch_size=64)
  print(res.mlflow_run_id)
  ```

### MLflow Registry and Model Loading

`[MLFLOW]` is a flat, client-only section. Its presence enables tracking; keep infrastructure endpoints in the environment:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_ARTIFACT_URI`

Legacy nested blocks such as `[MLFLOW.server]` / `[MLFLOW.client]` and TOML infra keys like
`tracking_uri` / `artifacts_destination` are rejected during validation.

`[MLFLOW].register_model` defaults to `false`.

- Registered model name defaults to the trained model class name.
- DLKit does **not** add default aliases or model-version tags.
- Aliases/tags are attached only when explicitly configured or set via API.
- `max_retries` controls transient MLflow client retry attempts.
- Dataset lineage is logged as a JSON artifact under `lineage/` plus run tags (`dataset_manifest_artifact`, `dataset_source_count`, `dataset_fingerprint`).

```python
from dlkit import (
    search_registered_models,
    list_model_versions,
    load_registered_model,
)

tracking_uri = "file:///tmp/mlruns"
model_name = "ConstantWidthFFNN"

registered = search_registered_models(model_name, tracking_uri=tracking_uri)
versions = list_model_versions(model_name, tracking_uri=tracking_uri)

# Alias-based loading (defaults to MLflow latest resolution)
latest = load_registered_model(model_name, alias="latest", tracking_uri=tracking_uri)

# Version-pinned loading
pinned = load_registered_model(model_name, version=versions[-1], tracking_uri=tracking_uri)
```

#### From TrainingResult to Register / Alias / Tag / Load

Use the training output to register and annotate models programmatically:

```python
import dlkit
import mlflow

settings = ...  # your GeneralSettings
result = dlkit.train(settings)

run_id = result.mlflow_run_id
if run_id is None:
    raise RuntimeError("MLflow run id missing from training result")

tracking_uri = result.mlflow_tracking_uri or mlflow.get_tracking_uri()
model_name = getattr(settings.MLFLOW, "registered_model_name", None) or type(
    result.model_state.model
).__name__

# Register from the run artifact (useful when register_model=false)
version_entity = dlkit.register_logged_model(
    model_name,
    run_id=run_id,
    artifact_path="model",
    tracking_uri=tracking_uri,
)
version = int(version_entity.version)

# Attach aliases and tags explicitly
dlkit.set_registered_model_alias(
    model_name,
    alias="dataset_A_latest",
    version=version,
    tracking_uri=tracking_uri,
)
dlkit.set_registered_model_version_tags(
    model_name,
    version=version,
    tags={"dataset": "dataset_A", "benchmark": "high_precision"},
    tracking_uri=tracking_uri,
)

# Load by alias (PyTorch flavor preferred, sklearn/pyfunc fallback in auto mode)
model = dlkit.load_registered_model(
    model_name,
    alias="dataset_A_latest",
    tracking_uri=tracking_uri,
    flavor="auto",  # "pytorch" | "sklearn" | "pyfunc"
)
```

You can also set aliases/tags declaratively in TOML:

```toml
[MLFLOW]
register_model = true
registered_model_name = "FFNN"
registered_model_aliases = ["dataset_A_latest", "benchmark_high_precision"]
registered_model_version_tags = { dataset = "dataset_A", benchmark = "high_precision" }
```

When you set `[MLFLOW].register_model = false`, DLKit still logs the model artifact under the run (`runs:/...`) and you can locate/load it with logged-model helpers:

```python
from dlkit import search_logged_models, load_logged_model

tracking_uri = "file:///tmp/mlruns"
results = search_logged_models(
    model_name="ConstantWidthFFNN",
    experiment_name="my_experiment",
    tracking_uri=tracking_uri,
)

latest = results[0]
model = load_logged_model(model_uri=latest.model_uri, tracking_uri=tracking_uri)
```

### Accessing Stacked Predictions from TrainingResult

`TrainingResult.stacked` concatenates prediction batches into a single `TensorDict`
with keys `predictions`, `targets`, and `latents` (cached on first access):

```python
result = dlkit.train(settings, epochs=10)
stacked = result.stacked  # TensorDict | None

if stacked is not None:
    predictions = stacked["predictions"]  # Tensor or nested TensorDict
    targets = stacked["targets"]
    latents = stacked["latents"]          # (N, 0) sentinel when absent
```

To get NumPy arrays, use `to_numpy()`:

```python
arrays = result.to_numpy()                          # all keys
pred_only = result.to_numpy("predictions")          # top-level key
target_y = result.to_numpy(("targets", "y"))        # nested key path
```

## Inference

DLKit loads models from checkpoints with automatic transform handling and precision inference. **No configuration files are required** - everything needed is extracted from the checkpoint metadata.

### Load Model Once, Use Many Times

```python
from dlkit import load_model
import torch

# Load checkpoint ONCE (expensive operation)
predictor = load_model(
  checkpoint_path="model.ckpt",
  device="auto",  # "auto", "cpu", "cuda", "mps"
  apply_transforms=True  # Apply saved transforms automatically
)

# Option 1: Use the model directly (full PyTorch control)
model = predictor.model  # Get the loaded PyTorch model
predictions = model(inputs)  # Standard PyTorch forward pass

# Option 2: Use predictor.predict() — mirrors model.forward() exactly
output = predictor.predict(x=torch.randn(32, 10))  # → Tensor

# Cleanup when done
predictor.unload()
```

**Why this is faster**: The checkpoint is loaded from disk once. All subsequent operations use the cached model in memory - no redundant I/O.

### Direct Model Access

The loaded model is a standard PyTorch module - use it however you want:

```python
predictor = load_model("model.ckpt", device="cuda")
model = predictor.model  # Standard nn.Module

# Standard PyTorch usage
with torch.no_grad():
    output = model(inputs)

# Access layers, parameters, state
for name, param in model.named_parameters():
    print(name, param.shape)

# Extract intermediate activations
features = model.encoder(inputs)

# Fine-tune or train further
optimizer = torch.optim.Adam(model.parameters())
```

### High-Level Predict API

`predict()` mirrors `model.forward()` — pass tensors as positional or keyword args:

```python
# Single input (most common)
output = predictor.predict(x=torch.randn(32, 10))       # → Tensor

# Multi-input (kwargs)
output = predictor.predict(x=x_tensor, edge_attr=ea)    # → Tensor

# Multi-input (positional)
output = predictor.predict(x_tensor, ea_tensor)          # → Tensor

# Multi-output model (e.g. VAE): returns tuple
recon, mu, logvar = predictor.predict(x=x_tensor)

# Without transforms (raw model output in normalized space)
predictor_raw = load_model("model.ckpt", apply_transforms=False)
output = predictor_raw.predict(x=normalized_inputs)
```

`apply_transforms=True` (default) automatically applies forward feature transforms
before the call and the inverse target transform after, so you always receive
predictions in the original data space.

### Context Manager (Automatic Cleanup)

For temporary usage:

```python
from dlkit import load_model

with load_model("model.ckpt", device="cuda") as predictor:
  model = predictor.model
  output = model(inputs)
# Automatic cleanup and GPU memory release
```

### Input Formats

`predict()` accepts any combination of positional and keyword `torch.Tensor` arguments,
exactly as you would call the underlying `model.forward()`:

```python
# Keyword arg (explicit, recommended)
output = predictor.predict(x=torch.randn(32, 10))

# Positional arg (same as above for single-input models)
output = predictor.predict(torch.randn(32, 10))

# Mixed positional + keyword (multi-input models)
output = predictor.predict(x_tensor, edge_attr=ea_tensor)
```

NumPy arrays are **not** auto-converted — call `torch.from_numpy(arr)` before passing.

## Transform Pipelines (Training → Inference)

DLKit ships a unified transform system that fits data transforms during training, persists them in checkpoints, and reapplies them automatically during inference. This gives you scikit-learn style `fit ↔ transform ↔ inverse_transform` semantics without writing glue code.

### Transform Workflow: Normalized Space Training

**Key Principle**: The model trains and computes loss in **normalized space** for better convergence and gradient stability.

**Training Flow:**
```
Raw Data → Forward Transform → Normalized Space → Model → Loss (in normalized space)
           features:  [0, 100]  →  [-1, 1]
           targets:   [0, 100]  →  [-1, 1]
           predictions: model outputs in normalized space
           loss: MSE(predictions_normalized, targets_normalized)
```

**Inference Flow:**
```
Raw Features → Forward Transform → Model → Inverse Transform → Original Space Predictions
```

### Declaring Transforms in Configuration

Attach transforms to features/targets in your TOML config via `TransformSettings`:

```toml
[DATASET]
features = [
  { name = "x", path = "data/features.npy", dtype = "float32", transforms = [
    { name = "MinMaxScaler", module_path = "dlkit.core.training.transforms.minmax", dim = 0 },
    { name = "SampleNormL2", module_path = "dlkit.core.training.transforms.sample_norm", eps = 1e-8 },
  ] }
]
targets = [
  { name = "y", path = "data/targets.npy", dtype = "float32", transforms = [
    { name = "PCA", module_path = "dlkit.core.training.transforms.pca", n_components = 8 },
  ] }
]
```

**Available Transforms:**

- **MinMaxScaler** (`dlkit.core.training.transforms.minmax`): Min-max normalization to [-1, 1] range (fittable, invertible)
- **StandardScaler** (`dlkit.core.training.transforms.standard`): Z-score normalization using mean/std (fittable, invertible)
- **PCA** (`dlkit.core.training.transforms.pca`): Principal component analysis for dimensionality reduction (fittable, invertible; online fitting currently requires pre-fitted state)
- **SampleNormL2** (`dlkit.core.training.transforms.sample_norm`): Per-sample L2 normalization (invertible only - does not require fitting)
- **Permutation** (`dlkit.core.training.transforms.permute`): Permute tensor dimensions
- **TensorSubset** (`dlkit.core.training.transforms.subset`): Extract tensor subsets
- **SpectralRadiusNorm** (`dlkit.core.training.transforms.spectral`): Spectral radius normalization for matrices

### How Transforms Are Fitted and Applied

**Fitting Phase** (`on_fit_start()` - before first training step):
1. Builds `TransformChain` per named entry (features and targets)
2. Fits each chain using a **streaming multi-pass dataloader flow** (no full `torch.cat` buffering)
3. Incremental transforms (`MinMaxScaler`, `StandardScaler`) accumulate fit state batch-by-batch
4. Unfitted non-incremental fittable transforms fail fast (current policy: `PCA` online fit rejected; TODO incremental PCA)
5. Stores fitted chains inside a `NamedBatchTransformer` with two `ModuleDict`s:
   - `_batch_transformer._feature_chains.<name>` - for input preprocessing
   - `_batch_transformer._target_chains.<name>` - for target normalization

**During Training/Validation/Test:**
```python
# Step-by-step flow:
features, targets = extract_from_batch(batch)           # Raw data

features = feature_transforms.forward(features)         # Raw → normalized
targets = target_transforms.forward(targets)            # Raw → normalized

predictions = model(features)                           # Predicts in normalized space

loss = compute_loss(predictions, targets)               # Both normalized - consistent scale!
```

**During Inference/Prediction:**
```python
features = feature_transforms.forward(features)         # Raw → normalized
predictions = model(features)                           # Predicts in normalized space
predictions = target_transforms.inverse(predictions)    # Normalized → raw (for user)
```

The `StandardLightningWrapper` handles all transform application automatically. Fitted transform state is persisted inside `_batch_transformer._feature_chains.<entry_name>.*` and `_batch_transformer._target_chains.<entry_name>.*` checkpoint keys.

### Persistence Guarantees

- Transforms are cached globally and written to the checkpoint alongside model weights.
- Saving via Lightning (`Trainer.save_checkpoint`) or a raw `state_dict()` preserves the fitted chains.
- Loading the wrapper or `load_model()` reconstructs the exact chain, including fitted parameters (e.g., min/max, mean/std, and any pre-fitted PCA components).

For checkpoint structure and state dict key patterns, see [`src/dlkit/core/models/wrappers/README.md`](src/dlkit/core/models/wrappers/README.md).

### Inference Behavior

`load_model(..., apply_transforms=True)` (default) will:

1. Apply **forward** feature transforms (raw data → normalized) before model forward pass
2. Model predicts in normalized space (same space it was trained in)
3. Apply **inverse** target transforms (normalized → raw) so predictions are in original units

This ensures:
- ✅ Model sees data in same normalized space as during training
- ✅ User receives predictions in original, interpretable units
- ✅ No manual transform handling required

For advanced workflows:

- **Raw inputs + default behavior** (recommended):

  ```python
  with dlkit.load_model("model.ckpt", apply_transforms=True) as predictor:
      predictions = predictor.predict(x=torch.from_numpy(raw_features))
      # predictions is already inverse-transformed — original data space
  ```

- **Pre-transformed tensors**: If you already normalized features, disable the automatic pass to avoid double application:

  ```python
  normalized = my_chain(raw_features)
  with dlkit.load_model("model.ckpt", apply_transforms=False) as predictor:
      logits = predictor.predict(x=normalized)
  ```

- **Manual control**: Use `load_transforms_from_checkpoint()` from `dlkit.interfaces.inference.transforms` to extract the fitted chains for custom serving stacks (e.g., streaming inference, Spark jobs).

These guarantees are covered by the integration tests in `tests/integration/test_transforms_persistence_and_inference.py`.

## Breaking Changes

### Transform API Changes (v2.1+)

**Breaking Change: `input_shape` Parameter Removed**

All transforms have been refactored to eliminate the redundant `input_shape` parameter. Transforms now integrate with the existing `shape_spec` system, providing a single source of truth for shape information.

**What Changed:**
- ❌ `Transform.__init__(input_shape=...)` - removed parameter
- ❌ `MinMaxScaler(dim=0, input_shape=(32, 64))` - old API
- ❌ `TransformChain(settings, input_shape=(32, 64))` - old API

**New Shape-Aware API:**
- ✅ Transforms support **automatic shape allocation** - no shape needed at construction
- ✅ Shape-aware transforms implement `IShapeAwareTransform` with `configure_shape()` method
- ✅ Shapes automatically inferred during `fit()` or provided via `shape_spec`
- ✅ `TransformChain` uses analytical shape inference (no dummy tensor execution)

**Migration Examples:**

```python
# ❌ OLD (removed)
from dlkit.core.training.transforms import MinMaxScaler, PCA
scaler = MinMaxScaler(dim=0, input_shape=(32, 64))
pca = PCA(n_components=10, input_shape=(32, 64))

# ✅ NEW (automatic allocation)
scaler = MinMaxScaler(dim=0)  # Shape inferred during fit()
pca = PCA(n_components=10)
scaler.fit(train_data)  # Automatically allocates buffers from data.shape
pca.fit(train_data)
```

**Configuration Files**: No changes needed - TOML configs remain the same. The shape summary is handled automatically by the training system.

**Why the Change?**

- **Single source of truth**: Eliminates redundant shape tracking between transforms and `shape_spec`
- **Better performance**: Analytical shape inference replaces dummy tensor execution
- **Cleaner API**: Transforms are truly composable without shape boilerplate
- **Architectural alignment**: Follows same pattern as `ShapeAwareModel` from neural network layer

**New Error Types:**
- `TransformNotFittedError` - raised when using unfitted transform (replaces generic `RuntimeError`)
- `ShapeMismatchError` - raised when shapes are incompatible
- `TransformChainError` - wraps errors from transform chains with context

---

### Inference API Changes (v2.0+)

**Removed APIs**:
- ❌ `infer()` function - replaced with `load_model()`
- ❌ `predict_with_config()` function
- ❌ `InferenceService` class
- ❌ `InferenceWorkflowSettings`
- ❌ `load_inference_settings()` function

**New Stateful Predictor API**:
- ✅ `load_model()` - primary inference API (loads once, predicts many)
- ✅ `CheckpointPredictor` - stateful predictor object
- ✅ `predict(*args, **kwargs) → Tensor | tuple[Tensor, ...]` — mirrors `model.forward()` exactly
- ✅ `validate_checkpoint()` - checkpoint validation utility
- ✅ `get_checkpoint_info()` - checkpoint metadata extraction

### Why the Change?

The old `infer()` function loaded the model from checkpoint **on every call**, causing hundreds of redundant loads during iterative workflows. The new stateful predictor architecture provides:

- **10-100x performance improvement** for multi-inference workflows
- Industry-standard API matching PyTorch, scikit-learn, and Hugging Face
- Clear model lifecycle management (load → predict × N → unload)
- No configuration files required - everything extracted from checkpoint

### Migration Guide

```python
# ❌ OLD (removed)
from dlkit import infer

for data in dataset:
  result = infer("model.ckpt", data)  # Reloaded 100+ times!
  process(result)

# ✅ NEW Option 1: Direct model access (standard PyTorch)
from dlkit import load_model

predictor = load_model("model.ckpt", device="cuda")
model = predictor.model  # Get the PyTorch model

with torch.no_grad():
  for data in dataset:
    output = model(data)  # Standard forward pass
    process(output)

predictor.unload()

# ✅ NEW Option 2: Use predictor.predict() (handles transforms)
predictor = load_model("model.ckpt", device="cuda", apply_transforms=True)

for data in dataset:
  output = predictor.predict(x=data)  # Transforms applied automatically
  process(output)                     # output is a Tensor (or tuple for multi-output)

predictor.unload()

# ✅ Context manager (automatic cleanup)
with load_model("model.ckpt") as predictor:
  model = predictor.model
  for data in dataset:
    output = model(data)
    process(output)
```

### Enhanced Checkpoint Format

DLKit automatically saves enhanced metadata with every checkpoint, enabling shape-free inference:

- **Model settings**: Complete model configuration for automatic reconstruction
- **Shape specifications**: Input/output tensor shapes inferred from training data
- **Transform configurations**: Feature and target transform chains for automatic application
- **Entry configurations**: Dataset entry mappings for proper data handling
- **Backward compatibility**: Legacy checkpoints still supported with fallback inference

This enhanced format eliminates the need to manually specify shapes or model configurations during inference.

## Minimal End-to-End Example

- This repo includes a tiny, ready-to-run example under `examples/minimal_e2e/`:
  - FlexibleDataset using two CSV files: `data/x.csv` (features) and `data/y.csv` (targets)
  - A small custom model `examples.minimal_e2e.model:SimpleNet` that accepts `shape` for auto-wiring
  - A minimal flattened TOML config at `examples/minimal_e2e/config.toml`

Run with the CLI
- `uv run dlkit train examples/minimal_e2e/config.toml --epochs 3`

Run with the Python API

```python
from dlkit.interfaces.api import train
from dlkit.tools.io import load_settings

cfg = load_settings("examples/minimal_e2e/config.toml")
result = train(cfg, epochs=3)
print(result.metrics)
```

Artifacts and logs default under `<root>/output/` where `<root>` comes from the environment (see below). In tests they go under `tests/artifacts/`.

## Environment-Based Paths

DLKit uses an environment-based path system with automatic resolution and optional `[PATHS]` section:

- Root directory: `DLKIT_ROOT_DIR` env var; defaults to current working directory.
- Config-based root: `[SESSION].root_dir` in your TOML can set the root per run; relative paths are resolved against the config file directory.
- Standard locations: available via `dlkit.tools.io.locations` (e.g., `locations.output()`, `locations.checkpoints_dir()`, `locations.mlruns_dir()`).
- Test runs: when running under pytest, outputs are routed to `tests/artifacts/` automatically.
- API overrides: `train(..., output_dir=..., data_dir=...)` set a temporary path context for that call without mutating settings.

### Optional PATHS Section

You can now optionally include a `[PATHS]` section for standardized paths with automatic resolution:

```toml
[PATHS]
# Common standardized paths (all optional)
matrix_path = "./data/matrix.txt"
output_dir = "./results"
checkpoint_path = "./checkpoints/model.ckpt"
data_dir = "./datasets"

# Custom paths are also supported via extras
custom_data = "./custom/data.csv"
```

All paths are resolved relative to SESSION.root_dir using DLKit's SecurePath system.

## Enhanced IO System

DLKit exposes two config-loading layers:

- High-level workflow loaders in `dlkit.tools.io` / `dlkit.tools.config`
- Low-level section readers and registry helpers in `dlkit.tools.io.config`

### Workflow Loaders

Use these for normal application code:

```python
from dlkit.tools.io import load_settings, load_sections

settings = load_settings("config.toml")  # TrainingWorkflowSettings
partial = load_sections("config.toml", ["MODEL", "DATASET"])
```

### Section-Level Loading

Use the low-level readers when you need explicit section models or registry-driven lookup:

```python
from dlkit.tools.config import SessionSettings
from dlkit.tools.io.config import (
    get_available_sections,
    load_section_config,
    load_sections_config,
)

sections = load_sections_config("config.toml", ["MODEL", "DATASET"])
model_config = sections["MODEL"]
dataset_config = sections["DATASET"]

session = load_section_config("config.toml", SessionSettings)
model = load_section_config("config.toml", section_name="MODEL")

available = get_available_sections("config.toml")
```

### Config Protocols

The low-level config API is documented by protocol contracts in `dlkit.tools.io.protocols`:

- `ConfigParser`: full-file and section-aware parsing
- `SectionExtractor`: extraction of named top-level sections
- `ConfigValidator[T]`: eager Pydantic validation for section payloads
- `PartialConfigReader`: high-level section reader contract

Current implementation details:

- `DLKitTomlSource` in `dlkit.tools.config.core.sources` reads TOML and preprocesses paths before section filtering.
- `load_sections_config()` and `load_section_config()` use the section-mapping registry in `dlkit.tools.io.config`.
- `register_section_mapping()` / `reset_section_mappings()` let custom models participate in registry-driven loading.
- Environment overrides follow `DLKIT_<SECTION>__<field>` and are merged via strict validated patching.

## URL Parsing Guidelines

When adding or updating validation utilities, **use Pydantic types for all URL parsing**. Do not import `urllib`, `requests`, `httpx`, or similar helpers inside the core datatypes packages (`src/dlkit/datatypes/`). The shared validators already expose helpers based on `pydantic_core.Url`; reuse those utilities so platform-specific quirks stay consistent across modules.

## Inference Transforms

DLKit automatically applies transforms during inference using configurations saved in checkpoint metadata. **No configuration files are needed** - transforms are automatically restored and applied.

### Configuration

Attach transforms to `[DATASET].features` and `[DATASET].targets` entries during training:

```toml
[DATASET]
name = "FlexibleDataset"
features = [
  { name = "x", path = "./data/features.npy", transforms = [
    { name = "MinMaxScaler", module_path = "dlkit.core.training.transforms.minmax", dim = 0 }
  ] }
]
targets = [
  { name = "y", path = "./data/targets.npy", transforms = [
    { name = "MinMaxScaler", module_path = "dlkit.core.training.transforms.minmax", dim = 0 }
  ] }
]
```

### Usage with Predictor

Control transform application during inference:

```python
from dlkit import load_model

# With transforms (default) — receives predictions in original data space
predictor = load_model("model.ckpt", apply_transforms=True)
output = predictor.predict(x=raw_data)

# Without transforms — receives raw model output (normalized space)
predictor = load_model("model.ckpt", apply_transforms=False)
output = predictor.predict(x=already_normalized_data)
```

### How It Works

- **Training**: Transform chains are fitted once on training data and saved in checkpoint metadata
- **Inference**: Predictors automatically load and apply saved transforms
  - Feature transforms applied before model forward pass
  - Inverse target transforms applied to predictions after model forward pass
- **Config-free**: No manual configuration needed - everything restored from checkpoint

### Notes

- Transform state is part of the enhanced checkpoint format
- For graph/timeseries wrappers with custom featurization, entry-based transforms are ignored to avoid conflicts
- Transform application can be toggled per-predictor instance

## Loss Pairing Rules

- Strict mapping: during train/val/test, DLKit pairs predictions to targets by name using a pipeline step (LossPairingStep).
- Single-target fallback: if there is exactly one target and one prediction, they are paired even if names differ.
- Autoencoders: when ``WrapperComponentSettings.is_autoencoder = true`` and the dataset has no explicit targets, features are used as targets (reconstruction) automatically.
- Errors are explicit: if a target key is missing a prediction (or a prediction is unexpected), an error lists the missing/unexpected keys and the available targets/predictions.

Model outputs
- For multi-target setups your model forward should return a dict with keys matching target names.
- For single-target setups returning a single tensor is fine (paired automatically).

## Advanced Loss Keyword Arguments

Losses and metrics can receive additional tensors from the batch, beyond the default
`(predictions, target)` pair.

- Mark context tensors with `model_input = false` so they are not passed to `model.forward()`.
- Use `loss_input` on `Feature`/`Target` entries for automatic loss-kwarg routing.
- Use `WRAPPER.loss_function.target_key` and `WRAPPER.loss_function.extra_inputs` for explicit routing.
- Metrics support the same pattern via `WRAPPER.metrics[*].target_key` and `extra_inputs`.

Detailed guide and edge-case behavior:
- [`src/dlkit/core/training/README.md`](src/dlkit/core/training/README.md)
- [`src/dlkit/core/models/wrappers/README.md`](src/dlkit/core/models/wrappers/README.md)

## Config Anatomy

- `[SESSION]`: high-level execution switches
  - `name`: run/study name.
  - `inference`: bool; when true, training/optuna are suspended.
- `[MODEL]`: model component settings
  - `name`, `module_path`: factory identifiers
  - `checkpoint`: optional path used in inference/resume
- `[DATASET]`: dataset configuration
  - `name`: e.g., `FlexibleDataset`
  - `features`/`targets`: lists of entries `{ name, path }`
  - `[DATASET.split]`: `val_ratio`, `test_ratio`, optional `filepath`
- `[DATAMODULE]`: dataloading settings
  - `name`: e.g., `InMemoryModule` (for array-like datasets)
  - `[DATAMODULE.dataloader]`: `batch_size`, `num_workers`, etc.
- `[TRAINING]`: training controls
  - `epochs`: convenient top-level epoch count
  - `[TRAINING.trainer]`: Lightning Trainer kwargs (e.g., `max_epochs`, `accelerator`)
  - `[TRAINING.optimizer]`: optimizer settings (e.g., `name`, `lr`)
- `[MLFLOW]`: experiment tracking
  - The section itself enables MLflow tracking; there is no separate `enabled` flag
  - `experiment_name`, `run_name`, `tags`, `register_model`, `max_retries`, optional `registered_model_name`, optional `registered_model_aliases`, optional `registered_model_version_tags`
  - `register_model` defaults to `false`; `enabled` is rejected during validation
  - Infra is env-only: `MLFLOW_TRACKING_URI`, `MLFLOW_ARTIFACT_URI`
  - Legacy nested sections are removed: `[MLFLOW.server]` and `[MLFLOW.client]` are invalid
  - Tracking URI resolution order: env URI -> `http://127.0.0.1:5000` when alive -> local sqlite fallback
- `[OPTUNA]`: hyperparameter optimization
  - `enabled`: bool; when true and selected, optimization runs
  - `n_trials`, `study_name`; optional `sampler`, `pruner`, `storage`
- `[EXTRAS]`: optional user-defined helpers (free-form), ignored by core

## Common Overrides

These runtime overrides can be applied via CLI flags or Python API keyword args. They are applied end-to-end by the override manager.

- checkpoint_path: overrides `[MODEL].checkpoint`
- output_dir: overrides the output base directory (path context)
- data_dir: overrides the input data directory (path context)
- epochs: overrides `[TRAINING].epochs` and `[TRAINING.trainer].max_epochs`
- batch_size: overrides `[DATAMODULE].batch_size` and `[DATAMODULE.dataloader].batch_size`
- learning_rate: overrides `[TRAINING.optimizer].lr`
- experiment_name: overrides `[MLFLOW].experiment_name`
- run_name: overrides `[MLFLOW].run_name`
- trials: overrides `[OPTUNA].n_trials`
- study_name: overrides `[OPTUNA].study_name`

Examples (CLI):
- `uv run dlkit train config.toml --epochs 20 --batch-size 64`
- `uv run dlkit train config_with_mlflow.toml --experiment-name MyExp --run-name Run1`
- `uv run dlkit optimize config_with_mlflow.toml --trials 50 --study-name Study`

Examples (Python):

```python
from dlkit.interfaces.api import train
from dlkit.tools.io import load_settings

# Load training config
cfg = load_settings("config.toml")
res = train(cfg, epochs=20, batch_size=64, learning_rate=1e-3)

# execute() is training/optimization only; for inference, use load_model instead
from dlkit import load_model
predictor = load_model("model.ckpt")
```

## Configuration Architecture: Mutable Settings with Validation

DLKit uses a **mutable settings architecture** with `frozen=False` and `validate_assignment=True`. This design choice enables efficient in-place updates while maintaining type safety through automatic validation on every assignment.

### Why Mutable Settings?

The architecture shifted from `frozen=True` (immutable) to `frozen=False` (mutable) to solve critical issues with value-based data entries:

**Problem with Immutable Approach:**
```python
# ❌ OLD: Serialization lost excluded fields
feature = ValueFeature(name="x", value=np.array([1, 2, 3]))  # value marked exclude=True
dataset = DatasetSettings(features=[feature])

# Immutable update required serialization:
updated = dataset.model_copy(update={"some_field": "value"})
# Problem: feature.value was excluded from serialization → LOST!
```

**Solution with Mutable Approach:**
```python
# ✅ NEW: Direct mutation preserves object identity
feature = ValueFeature(name="x", value=np.array([1, 2, 3]))
dataset = DatasetSettings(features=[feature])

# Mutable update preserves the original object:
update_settings(dataset, {"some_field": "value"})
# feature.value preserved (same object reference, no serialization)
```

### Key Benefits

1. **Preserves excluded fields**: No serialization means `ValueFeature.value` (marked `exclude=True`) is never lost
2. **Type safety maintained**: `validate_assignment=True` ensures every `setattr()` is type-checked
3. **Efficient updates**: No deep copying overhead for large nested structures
4. **Predictable behavior**: Same object identity after updates (important for cross-references)

### Using `update_settings()` for Configuration Updates

The `update_settings()` function provides a clean API for mutating nested settings in-place:

```python
from dlkit.tools.config.core.updater import update_settings
from dlkit.tools.config import load_settings

# Load existing config
config = load_settings("config.toml")

# Add features to existing dataset configuration
from dlkit.tools.config.data_entries import Feature

new_features = [
    Feature(name="x3", path="/data/feature3.npy"),
    Feature(name="x4", value=np.random.randn(100, 10))  # in-memory
]

# Update dataset by adding new features (mutable, in-place)
update_settings(config.DATASET, {
    "features": config.DATASET.features + tuple(new_features)
})

# Multiple nested updates in one call
update_settings(config, {
    "TRAINING": {
        "epochs": 100,
        "optimizer": {"lr": 0.001}
    },
    "DATAMODULE": {
        "dataloader": {"batch_size": 64}
    }
})

# Returns same object (mutated in-place)
assert id(config) == id(update_settings(config, {...}))
```

### Common Update Patterns

**1. Adding features to dataset:**
```python
# Load config
config = load_settings("config.toml")

# Create new features
new_feature = Feature(name="z", path="data/z.npy")

# Add to existing features (immutable tuple concatenation)
update_settings(config.DATASET, {
    "features": config.DATASET.features + (new_feature,)
})
```

**2. Replacing entire sections:**
```python
from dlkit.tools.config import DatasetSettings
from dlkit.tools.config.data_entries import Feature, Target

# Replace entire DATASET section
update_settings(config, {
    "DATASET": DatasetSettings(
        name="FlexibleDataset",
        features=(Feature(name="x", path="new_data.npy"),),
        targets=(Target(name="y", path="labels.npy"),)
    )
})
```

**3. Deep nested updates:**
```python
# Update deeply nested fields without replacing parent objects
update_settings(config, {
    "TRAINING": {
        "optimizer": {
            "lr": 0.001,
            "weight_decay": 1e-5
        }
    }
})
# Only lr and weight_decay are updated, other optimizer fields preserved
```

**4. Updating with in-memory arrays:**
```python
import numpy as np

# Inject in-memory data (zero file I/O)
features = np.random.randn(1000, 20).astype(np.float32)
update_settings(config.DATASET, {
    "features": (Feature(name="x_mem", value=features),)
})
```

### Notes

- **In-place mutation**: `update_settings()` mutates the input object and returns it
- **Validation**: Type validation happens automatically via `validate_assignment=True`
- **Nested merging**: Updates are deep-merged - only specified fields are overwritten
- **Object identity**: Same object before and after update (no copying)
- **No serialization**: Preserves object references and excluded fields

## Programmatic Configuration (Hybrid TOML + Python)

DLKit supports a powerful hybrid pattern combining TOML configuration with programmatic section injection. This is ideal for API workflows, experiments, and dynamic data scenarios where some configuration comes from files while other parts are generated at runtime.

### When to Use This Pattern

- **API Workflows**: Accepting data from HTTP requests, databases, or other sources
- **Experiments**: Iterating over datasets programmatically while keeping base config stable
- **Testing**: Injecting test data without creating temporary TOML files
- **Dynamic Paths**: Generating data paths based on runtime conditions
- **Parameterized Runs**: Sweeping over different dataset/model combinations programmatically

### Basic Pattern

The workflow consists of four steps (all eager validation; optional sections are whole modules):

1. **Load partial config** from TOML (required sections only, e.g., `SESSION`, `TRAINING`)
2. **Inject whole sections programmatically** using Pydantic's `model_copy(update=...)`
3. **Validate completeness** before building components
4. **Build and execute** using standard API

### Example 1: Injecting Dataset Configuration

Create a minimal TOML with only training settings:

```toml
# base_config.toml - Only defines training parameters
[SESSION]
name = "my_experiment"
seed = 42
precision = "float32"

[MODEL]
name = "examples.minimal_e2e.model.SimpleNet"

[TRAINING]
[TRAINING.trainer]
max_epochs = 100
accelerator = "auto"

[TRAINING.optimizer]
name = "Adam"
lr = 0.001
```

Then inject the dataset section programmatically:

```python
from dlkit.tools.io.config import load_training_config_eager
from dlkit.tools.config import DatasetSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.validators import validate_training_config_complete
from dlkit.runtime.workflows.factories.build_factory import BuildFactory

# 1. Load partial config from TOML
config = load_training_config_eager("base_config.toml")

# 2. Inject DATASET section programmatically
config = config.model_copy(update={
    "DATASET": DatasetSettings(
        name="FlexibleDataset",
        features=(
            Feature(name="x1", path="/data/features1.npy"),
            Feature(name="x2", path="/data/features2.npy"),
        ),
        targets=(
            Target(name="y", path="/data/labels.npy"),
        )
    )
})

# 3. Validate completeness before building
validate_training_config_complete(config)

# 4. Build and train
components = BuildFactory().build_components(config)
```

### Example 2: In-Memory Array Injection

For testing or API workflows, inject raw arrays directly without file I/O:

```python
import numpy as np
from dlkit.tools.io.config import load_training_config_eager
from dlkit.tools.config import DatasetSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.interfaces.api import train

# Load base config
config = load_training_config_eager("base_config.toml")

# Generate or receive data from API/database
features = np.random.randn(1000, 10).astype(np.float32)
labels = np.random.randn(1000, 1).astype(np.float32)

# Inject dataset with in-memory arrays (zero file I/O!)
config = config.model_copy(update={
    "DATASET": DatasetSettings(
        name="FlexibleDataset",
        features=(Feature(name="x", value=features),),
        targets=(Target(name="y", value=labels),)
    )
})

# Train directly on in-memory data
result = train(config, epochs=10)
```

### Example 3: Programmatic Model Injection

Inject model configuration dynamically:

```python
from dlkit.tools.config import ModelComponentSettings

config = load_training_config_eager("base_config.toml")

# Inject model configuration
config = config.model_copy(update={
    "MODEL": ModelComponentSettings(
        name="MyCustomModel",
        module_path="my_models.networks",
        hyperparams={"hidden_dim": 128, "num_layers": 3}
    )
})
```

### Example 4: Experiment Loop

Iterate over multiple datasets while keeping training config stable:

```python
from pathlib import Path
from dlkit.tools.io.config import load_training_config_eager
from dlkit.tools.config import DatasetSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.config.validators import validate_training_config_complete
from dlkit.interfaces.api import train

datasets = [
    ("experiment_1", Path("/data/exp1/features.npy"), Path("/data/exp1/labels.npy")),
    ("experiment_2", Path("/data/exp2/features.npy"), Path("/data/exp2/labels.npy")),
    ("experiment_3", Path("/data/exp3/features.npy"), Path("/data/exp3/labels.npy")),
]

base_config = load_training_config_eager("base_config.toml")

for exp_name, feature_path, label_path in datasets:
    # Inject dataset for this experiment
    config = base_config.model_copy(update={
        "SESSION": base_config.SESSION.model_copy(update={"name": exp_name}),
        "DATASET": DatasetSettings(
            name="FlexibleDataset",
            features=(Feature(name="x", path=str(feature_path)),),
            targets=(Target(name="y", path=str(label_path)),)
        )
    })

    # Validate and train
    validate_training_config_complete(config)
    result = train(config, epochs=50)
    print(f"{exp_name}: {result.metrics}")
```

### Benefits

- **Separation of concerns**: Static training parameters in TOML, dynamic data in Python
- **Type safety**: Full Pydantic validation on injected sections
- **Fail-fast**: Eager validation catches errors immediately
- **Flexibility**: Mix file-based and in-memory data as needed
- **Testability**: No temporary files needed for testing
- **Auditability**: Base config remains unchanged, variations are explicit in code

### Notes

- `load_training_config_eager()` validates sections **present in TOML** immediately
- Missing sections (like `DATASET`, `DATAMODULE`) can be injected later
- Always call `validate_training_config_complete()` before `BuildFactory.build_components()`
- Use `model_copy(update=...)` to create modified configs (Pydantic immutability)
- In-memory arrays use `.value` attribute (XOR with `.path` - enforced by Pydantic)

## Template and Config Utilities

- Load specific configuration sections (partial loading)
  - Python:
    ```python
    from dlkit.tools.io import load_settings, load_sections

    # Load full training config
    cfg = load_settings("config.toml")

    # Partial loading: only specific sections for custom workflows
    settings = load_sections("config.toml", ["MODEL", "DATASET"])

    # Partial loading with strict validation (all sections must exist)
    strict_cfg = load_sections("config.toml", ["MODEL", "DATASET"], strict=True)
    ```

- Write a config programmatically to TOML
  - Python:
    ```python
    from pathlib import Path
    from dlkit.tools.io.config import write_config
    from dlkit.tools.config import GeneralSettings

    cfg = GeneralSettings(
        TRAINING={"trainer": {"default_root_dir": "work"}},
        EXTRAS={"note": "example"},
    )
    write_config(cfg, Path("config.toml"))
    ```

- Sync templates/examples
  - CLI: `uv run dlkit config sync-templates --write`
  - Check drift (CI/local): `uv run dlkit config sync-templates --check`
  - Script (same source): `python scripts/sync_templates.py --write`

Notes
- Paths: resolved against `DLKIT_ROOT_DIR` (or CWD). Standard subdirs are under `<root>/output`. No directories are created during config load; provisioning happens at runtime.
- The CLI templates are richly commented; feel free to prune comments in your production configs.


## User Registries (Custom Components)

Register your own components (models, datasets, losses, metrics, datamodules) and reference them by name in configs, or force their use without editing configs.

- Public API: `from dlkit.tooling.registry import register_model, register_dataset, register_loss, register_metric, register_datamodule`.
- Decorators support: auto-name (from `__name__`), `aliases`, `overwrite`, and `use=True` to force selection.
- Resolution order: forced selection (`use=True`) → registered name/alias → import by dotted path (for dlkit or third-party).

Examples
- Model: `@register_model(use=True)` on `class MyNet(nn.Module): ...` forces usage; config can omit `[MODEL].name`.
- Dataset: `@register_dataset(name="ToyDataset")` then set `[DATASET].name = "ToyDataset"`.
- Loss: `@register_loss(name="mae", aliases=["l1"])` and reference `mae`.
- Metric/Datamodule: identical pattern via `register_metric` / `register_datamodule`.



## Contributing

Contributions are welcome! Any suggestions or bug reports can be raised [here](https://github.com/constatza/dlkit/issues).
