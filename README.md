# DLKit

DLKit provides a thin, typed workflow layer on top of PyTorch and Lightning with:
- Flattened, typed configuration models (Pydantic) and a single TOML loader
- Strategy-based training (vanilla, MLflow) and hyperparameter optimization (Optuna)
- Simple dataset/datamodule factories with shape inference
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

## Quick Start (CLI)


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
# Computation precision preset (e.g., '32', '16-mixed')
precision = "32"

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

[EXTRAS]
# Optional user-defined helpers (ignored by core)
example_key = "value"
```

## Training and Optimization

- **Training**: Use `dlkit train` command (vanilla or with MLflow tracking)
- **Optimization**: Use `dlkit optimize` command for hyperparameter tuning with Optuna
- **MLflow tracking**: Add `--mlflow` flag to either command to enable tracking

Examples:
- CLI (vanilla training): `uv run dlkit train config.toml --epochs 20`
- CLI (training with MLflow): `uv run dlkit train config.toml --mlflow --epochs 20`
- CLI (optuna optimization): `uv run dlkit optimize config.toml --trials 50`
- CLI (optuna with MLflow): `uv run dlkit optimize config.toml --trials 50 --mlflow`
- Python (mlflow):
  ```python
  from dlkit.interfaces.api import train
  from dlkit.tools.io import load_settings
  cfg = load_settings("config.toml")
  res = train(cfg, mlflow=True, epochs=20, batch_size=64)
  ```

## Inference

DLKit loads models from checkpoints with automatic transform handling and precision inference. **No configuration files are required** - everything needed is extracted from the checkpoint metadata.

### Load Model Once, Use Many Times

```python
from dlkit import load_model

# Load checkpoint ONCE (expensive operation)
predictor = load_model(
  checkpoint_path="model.ckpt",
  device="cuda",  # "auto", "cpu", "cuda", "mps"
  apply_transforms=True  # Apply saved transforms automatically
)

# Option 1: Use the model directly (full PyTorch control)
model = predictor.model  # Get the loaded PyTorch model
predictions = model(inputs)  # Standard PyTorch forward pass

# Option 2: Use predictor.predict() (handles transforms automatically)
result = predictor.predict({"x": torch.randn(32, 10)})

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

If you want automatic transform handling, use `predictor.predict()`:

```python
# Transforms applied automatically (if saved in checkpoint)
result = predictor.predict({"x": torch.randn(32, 10)})
predictions = result.predictions  # Already inverse-transformed

# Without transforms (raw model output)
predictor_raw = load_model("model.ckpt", apply_transforms=False)
result = predictor_raw.predict({"x": normalized_inputs})
```

### Context Manager (Automatic Cleanup)

For temporary usage:

```python
from dlkit import load_model

with load_model("model.ckpt", device="cuda") as predictor:
  model = predictor.model
  output = model(inputs)
# Automatic cleanup and GPU memory release
```

### Flexible Input Formats

The `predict()` method accepts multiple input formats:

```python
# Tensors
result = predictor.predict({"x": torch.randn(32, 10)})

# NumPy arrays (auto-converted)
result = predictor.predict({"x": np.random.randn(32, 10)})

# Single tensor (auto-wrapped)
result = predictor.predict(torch.randn(32, 10))
```

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
[DATASET.entries.x]
path = "data/features.npy"
dtype = "float32"
[[DATASET.entries.x.transforms]]
name = "MinMaxScaler"
module_path = "dlkit.core.training.transforms.minmax"
dim = 0

# Per-sample L2 normalization (no fitting required)
[[DATASET.entries.x.transforms]]
name = "SampleNormL2"
module_path = "dlkit.core.training.transforms.sample_norm"
eps = 1e-8

[DATASET.entries.y]
path = "data/targets.npy"
dtype = "float32"
[[DATASET.entries.y.transforms]]
name = "PCA"
module_path = "dlkit.core.training.transforms.pca"
n_components = 8
```

**Available Transforms:**

- **MinMaxScaler** (`dlkit.core.training.transforms.minmax`): Min-max normalization to [-1, 1] range (fittable, invertible)
- **StandardScaler** (`dlkit.core.training.transforms.standard`): Z-score normalization using mean/std (fittable, invertible)
- **PCA** (`dlkit.core.training.transforms.pca`): Principal component analysis for dimensionality reduction (fittable, invertible)
- **SampleNormL2** (`dlkit.core.training.transforms.sample_norm`): Per-sample L2 normalization (invertible only - does not require fitting)
- **Permutation** (`dlkit.core.training.transforms.permute`): Permute tensor dimensions
- **TensorSubset** (`dlkit.core.training.transforms.subset`): Extract tensor subsets
- **SpectralRadiusNorm** (`dlkit.core.training.transforms.spectral`): Spectral radius normalization for matrices

### How Transforms Are Fitted and Applied

**Fitting Phase** (`on_fit_start()` - before first training step):
1. Accumulates **entire training dataset** across all batches
2. Builds `TransformChain` per entry (features and targets)
3. Fits chains globally on raw training data
4. Stores fitted chains in two separate `ModuleDict`s:
   - `fitted_feature_transforms` - for input preprocessing
   - `fitted_target_transforms` - for target normalization

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

The `StandardLightningWrapper` handles all transform application automatically. Fitted transform state is persisted inside `fitted_feature_transforms.*` and `fitted_target_transforms.*` checkpoint keys.

### Persistence Guarantees

- Transforms are cached globally and written to the checkpoint alongside model weights.
- Saving via Lightning (`Trainer.save_checkpoint`) or a raw `state_dict()` preserves the fitted chains.
- Loading the wrapper or `load_model()` reconstructs the exact chain, including running stats (e.g., min/max, PCA components).

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
      result = predictor.predict({"x": torch.from_numpy(raw_features)})
      predictions = result.predictions["y"]  # already inverse-transformed
  ```

- **Pre-transformed tensors**: If you already normalized features, disable the automatic pass to avoid double application:

  ```python
  normalized = my_chain({"x": raw_features})["x"]
  with dlkit.load_model("model.ckpt", apply_transforms=False) as predictor:
      logits = predictor.predict({"x": normalized})
  ```

- **Manual control**: Use `TransformChainExecutor.from_checkpoint("model.ckpt")` to pull out the fitted chains for custom serving stacks (e.g., streaming inference, Spark jobs). Apply `apply_feature_transforms()` or `apply_inverse_target_transforms()` against your own tensors whenever needed.

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
  result = predictor.predict(data)  # Transforms applied automatically
  process(result.predictions)

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

DLKit provides a comprehensive IO system with efficient section-level config loading and protocol-based design:

### Partial Config Loading (Section-Level)

Load only specific config sections with eager validation; missing sections are supported only at whole-section granularity (e.g., omit `DATASET`/`DATAMODULE`/`MODEL` and inject later):

```python
from dlkit.tools.io.config import load_sections_config, load_section_config

# Load multiple sections efficiently (eager validation)
sections = load_sections_config("config.toml", ["MODEL", "DATASET"])
model_config = sections["MODEL"]
dataset_config = sections["DATASET"]

# Load single section
model_section = load_section_config("config.toml", "MODEL")

# Check available sections without full parsing
from dlkit.tools.io.config import get_available_sections
available = get_available_sections("config.toml")
```

### IO Protocols and Parsers

The IO system follows SOLID principles with configurable parsers:

- **`ConfigParser`**: Strategy pattern for different parsing approaches
- **`PartialTOMLParser`**: Efficient section extraction for large configs
- **`SectionExtractor`**: Clean section extraction from parsed data
- **`ConfigValidator`**: Pydantic-based validation with proper error handling

### IO Module Structure

- **`dlkit.tools.io.config`**: Config loading, validation, and writing
- **`dlkit.tools.io.locations`**: Centralized path policy and standard locations
- **`dlkit.tools.io.provisioning`**: Explicit directory creation at runtime
- **`dlkit.tools.io.protocols`**: Protocol definitions for parsers and validators
- **`dlkit.tools.io.parsers`**: Concrete parser implementations

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

# With transforms (default)
predictor = load_model("model.ckpt", apply_transforms=True)
result = predictor.predict({"x": raw_data})  # Transforms applied automatically

# Without transforms (raw data)
predictor = load_model("model.ckpt", apply_transforms=False)
result = predictor.predict({"x": already_normalized_data})
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
  - `enabled`: bool; when true, MLflow is configured/used
  - `[MLFLOW.server]`: `scheme`, `host`, `port`, optional storage URIs
  - `[MLFLOW.client]`: `tracking_uri` (auto from server), `experiment_name`, `run_name`, `register_model`
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
- mlflow_host: overrides `[MLFLOW.server].host`
- mlflow_port: overrides `[MLFLOW.server].port`
- experiment_name: overrides `[MLFLOW.client].experiment_name`
- run_name: overrides `[MLFLOW.client].run_name`
- trials: overrides `[OPTUNA].n_trials`
- study_name: overrides `[OPTUNA].study_name`

Examples (CLI):
- `uv run dlkit train config.toml --epochs 20 --batch-size 64`
- `uv run dlkit train config.toml --mlflow --experiment-name MyExp --run-name Run1`
- `uv run dlkit optimize config.toml --trials 50 --study-name Study --mlflow`

Examples (Python):

```python
from dlkit.interfaces.api import train
from dlkit.tools.io import load_settings

# Load training config
cfg = load_settings("config.toml")
res = train(cfg, epochs=20, batch_size=64, learning_rate=1e-3)

# For inference, use load_model instead
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
precision = "32"

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
