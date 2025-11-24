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
  from dlkit.tools.config import load_settings
  cfg = load_settings("config.toml", inference=False)
  res = train(cfg, strategy="mlflow", epochs=20, batch_size=64)
  ```

## Inference

DLKit provides an industry-standard inference system with **stateful predictors** that load models once and reuse them for multiple predictions. **No configuration files are required** - everything needed is extracted from the checkpoint using enhanced metadata.

### Recommended Method: Stateful Predictors

The **primary and most efficient approach** uses stateful predictor objects:

```python
from dlkit import load_predictor

# Load model ONCE
predictor = load_predictor(
    checkpoint_path="model.ckpt",
    device="cuda",              # "auto", "cpu", "cuda", "mps"
    batch_size=32,
    apply_transforms=True       # Automatically applies saved transforms
)

# Predict MANY times (no reloading!)
result1 = predictor.predict({"x": torch.randn(32, 10)})
result2 = predictor.predict({"x": torch.randn(32, 10)})
result3 = predictor.predict({"x": torch.randn(32, 10)})

# Cleanup when done
predictor.unload()
```

**Why this is faster**: The model is loaded from disk once and cached in memory. Each `predict()` call performs only a fast forward pass without any checkpoint I/O.

### Context Manager (Automatic Cleanup)

For one-shot or temporary inference, use the context manager:

```python
from dlkit import load_predictor

# Model loads on entry, auto-cleans on exit
with load_predictor("model.ckpt", device="cuda") as predictor:
    result = predictor.predict({"x": torch.randn(32, 10)})
# Automatic cleanup - no need to call unload()
```

### Config-Based Batch Inference

For inference on full datasets defined in configuration files:

```python
from dlkit import load_predictor

predictor = load_predictor("model.ckpt")

# Predict on dataset/split from config (yields results per batch)
for batch_result in predictor.predict_from_config("config.toml"):
    predictions = batch_result.predictions
    process_batch(predictions)

predictor.unload()
```

### Flexible Input Formats

Predictors accept multiple input formats:

```python
# Tensors
result = predictor.predict({"x": torch.randn(32, 10)})

# NumPy arrays
result = predictor.predict({"x": np.random.randn(32, 10)})

# File paths
result = predictor.predict({"features": "data/test_features.csv"})

# Mixed formats
result = predictor.predict({
    "x": torch.randn(32, 10),
    "metadata": "data/metadata.npy"
})
```

## Transform Pipelines (Training → Inference)

DLKit ships a unified transform system that fits data transforms during training, persists them in checkpoints, and reapplies them automatically during inference. This gives you scikit-learn style `fit ↔ transform ↔ inverse_transform` semantics without writing glue code.

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

During training the `StandardLightningWrapper` builds a `TransformChain` per entry, fits it once (across the entire training split), and stores the fitted chain inside `fitted_transforms.*` in the checkpoint.

### Persistence Guarantees

- Transforms are cached globally and written to the checkpoint alongside model weights.
- Saving via Lightning (`Trainer.save_checkpoint`) or a raw `state_dict()` preserves the fitted chains.
- Loading the wrapper or `load_predictor()` reconstructs the exact chain, including running stats (e.g., min/max, PCA components).

### Inference Behavior

`load_predictor(..., apply_transforms=True)` (default) will:

1. Apply feature transforms before forwarding through the model.
2. Apply inverse target transforms on the outputs so predictions land back in the user’s original space.

For advanced workflows:

- **Raw inputs + default behavior** (recommended):

  ```python
  with dlkit.load_predictor("model.ckpt", apply_transforms=True) as predictor:
      result = predictor.predict({"x": torch.from_numpy(raw_features)})
      predictions = result.predictions["y"]  # already inverse-transformed
  ```

- **Pre-transformed tensors**: If you already normalized features, disable the automatic pass to avoid double application:

  ```python
  normalized = my_chain({"x": raw_features})["x"]
  with dlkit.load_predictor("model.ckpt", apply_transforms=False) as predictor:
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
- ✅ Transforms support **lazy shape allocation** - no shape needed at construction
- ✅ Shape-aware transforms implement `IShapeAwareTransform` with `configure_shape()` method
- ✅ Shapes automatically inferred during `fit()` or provided via `shape_spec`
- ✅ `TransformChain` uses analytical shape inference (no dummy tensor execution)

**Migration Examples:**

```python
# ❌ OLD (removed)
from dlkit.core.training.transforms import MinMaxScaler, PCA
scaler = MinMaxScaler(dim=0, input_shape=(32, 64))
pca = PCA(n_components=10, input_shape=(32, 64))

# ✅ NEW (lazy allocation)
scaler = MinMaxScaler(dim=0)  # Shape inferred during fit()
pca = PCA(n_components=10)
scaler.fit(train_data)  # Automatically allocates buffers from data.shape
pca.fit(train_data)

# ✅ NEW (with shape_spec - used internally by pipelines)
from dlkit.core.shape_specs import create_shape_spec
shape_spec = create_shape_spec({"features": (32, 64)})
scaler = MinMaxScaler(dim=0)
scaler.configure_shape(shape_spec, "features")  # Eager allocation
```

**Configuration Files**: No changes needed - TOML configs remain the same. The `shape_spec` is handled automatically by the training system.

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
- ❌ `infer()` function - replaced with `load_predictor()`
- ❌ `predict_with_config()` function
- ❌ `InferenceService` class
- ❌ `InferenceWorkflowSettings`
- ❌ `load_inference_settings()` function

**New Stateful Predictor API**:
- ✅ `load_predictor()` - primary inference API (loads once, predicts many)
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

# ✅ NEW (efficient)
from dlkit import load_predictor

# Load model ONCE
predictor = load_predictor("model.ckpt", device="cuda")

# Predict many times (no reloading!)
for data in dataset:
    result = predictor.predict(data)  # Fast forward pass only
    process(result)

predictor.unload()

# ✅ Or with context manager (automatic cleanup)
with load_predictor("model.ckpt") as predictor:
    for data in dataset:
        result = predictor.predict(data)
        process(result)
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
from dlkit.tools.config import load_settings

cfg = load_settings("examples/minimal_e2e/config.toml")  # defaults to training
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

DLKit now provides a comprehensive IO system with efficient partial config loading and protocol-based design:

### Partial Config Loading

Load only specific config sections without parsing the entire file:

```python
from dlkit.tools.io.config import load_sections_config, load_section_config

# Load multiple sections efficiently
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
from dlkit import load_predictor

# With transforms (default)
predictor = load_predictor("model.ckpt", apply_transforms=True)
result = predictor.predict({"x": raw_data})  # Transforms applied automatically

# Without transforms (raw data)
predictor = load_predictor("model.ckpt", apply_transforms=False)
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
from dlkit.tools.config import load_settings

# Recommended: boolean-based loading (consistent with SESSION.inference)
cfg = load_settings("config.toml", inference=False)  # training mode
res = train(cfg, epochs=20, batch_size=64, learning_rate=1e-3)

# Inference mode
cfg = load_settings("config.toml", inference=True)   # inference mode

# Default (training mode)
cfg = load_settings("config.toml")  # defaults to training

# Alternative: class-based loading
from dlkit.tools.config import GeneralSettings
cfg = GeneralSettings.from_toml_file("config.toml")
res = train(cfg, epochs=20, batch_size=64, learning_rate=1e-3)
```

## Template and Config Utilities

- Load specific configuration sections (partial loading)
  - Python:
    ```python
    from dlkit.tools.config import load_settings, load_sections

    # Main API: boolean-based loading (consistent with SESSION.inference)
    training_cfg = load_settings("config.toml", inference=False)  # or just load_settings("config.toml")
    inference_cfg = load_settings("config.toml", inference=True)

    # Partial loading: only specific sections for custom workflows
    settings = load_sections("config.toml", ["MODEL", "DATASET"])

    # Partial loading with strict validation (all sections must exist)
    strict_cfg = load_sections("config.toml", ["MODEL", "DATASET"], strict=True)

    # Efficient section-based loading (new IO system)
    from dlkit.tools.io.config import load_sections_config, load_section_config
    sections = load_sections_config("config.toml", ["MODEL", "DATASET"])
    model_only = load_section_config("config.toml", "MODEL")
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
