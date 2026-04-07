# DLKit

A typed, configuration-driven training and inference toolkit built on
[PyTorch](https://pytorch.org/) and
[Lightning](https://lightning.ai/docs/pytorch/stable/).

## Table of Contents

- [What is DLKit](#what-is-dlkit)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [MLflow Tracking](#mlflow-tracking)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Data Formats](#data-formats)
- [Transforms](#transforms)
- [Advanced: Programmatic Configuration](#advanced-programmatic-configuration)
- [Custom Components](#custom-components)
- [Environment and Paths](#environment-and-paths)
- [Contributing](#contributing)

---

## What is DLKit

DLKit wraps [PyTorch](https://pytorch.org/) and
[Lightning](https://lightning.ai/docs/pytorch/stable/) with a typed TOML configuration
layer. You describe your experiment in a single config file; DLKit wires up the
dataset, model, optimizer, trainer, transforms, and optionally
[MLflow](https://mlflow.org/) tracking or
[Optuna](https://optuna.org/) hyperparameter search.

Key features:

- **Single TOML config** — flat, richly commented, Pydantic-validated
- **Load-once inference** — `load_model("model.ckpt")` reconstructs the model and all
  fitted transforms from the checkpoint; no config file needed at inference time
- **Strategy-based training** — vanilla, MLflow-tracked, or Optuna-optimized with the same config
- **Flexible dataset** — NumPy, PyTorch, CSV, and sparse-pack data out of the box
- **Composable transforms** — MinMaxScaler, StandardScaler, PCA, and more; fitted during
  training and automatically applied at inference

---

## Installation

Requires **Python 3.14+**.

Install [uv](https://docs.astral.sh/uv/) first, then sync with an explicit accelerator extra:

```bash
uv sync --extra cpu
```

CUDA builds are only published for Linux and Windows. Use one of the explicit CUDA extras when you want a GPU wheel:

```bash
uv sync --extra cu128
uv sync --extra cu130
```

DLKit does not define a default PyTorch backend. One of `cpu`, `cu128`, or `cu130` must be selected for project installs and lockfile exports.

---

## Quick Start

The repository ships with a runnable minimal example under `examples/minimal_e2e/`.

### Train

```bash
# CLI
uv run dlkit train examples/minimal_e2e/config.toml --epochs 3
```

```python
# Python API
from dlkit.infrastructure.config import load_settings
from dlkit.interfaces.api import train

cfg = load_settings("examples/minimal_e2e/config.toml")
result = train(cfg, epochs=3)

print(result.metrics)  # val_loss, etc.
print(result.artifacts)  # {"best_checkpoint": Path(...)}
```

### Predict from checkpoint

```python
from dlkit import load_model
import torch

# Load once (reads checkpoint, reconstructs model + transforms)
predictor = load_model("checkpoints/model.ckpt", device="auto")

# Predict many times — no reloading
for batch in my_data:
    output = predictor.predict(x=batch)

predictor.unload()
```

---

## Configuration

DLKit uses a single flat TOML file. Generate a commented template with:

```bash
uv run dlkit config create --output config.toml
```

### Section overview

| Section | Required | Purpose |
|---------|----------|---------|
| `[SESSION]` | yes | Run name, seed, precision |
| `[MODEL]` | yes | Model class and hyperparameters |
| `[TRAINING]` | yes | Trainer, optimizer, scheduler settings |
| `[DATAMODULE]` | yes | DataModule class and dataloader settings |
| `[DATASET]` | yes | Feature/target file paths and entry configuration |
| `[MLFLOW]` | no | Experiment tracking — presence enables MLflow |
| `[OPTUNA]` | no | Hyperparameter search settings |
| `[EXTRAS]` | no | Free-form user-defined values |

### Minimal annotated config

```toml
[SESSION]
name = "my_run"        # Used in logs and MLflow
seed = 42
precision = "float32"  # "float32" | "float64" | "mixed16" | "bf16" | etc.

[MODEL]
name = "my_package.models.MyNet"  # Dotted class path or registry alias

[TRAINING]

[TRAINING.trainer]
max_epochs = 100
accelerator = "auto"   # "cpu" | "gpu" | "auto"

[TRAINING.optimizer]
name = "Adam"
lr = 1e-3

[DATAMODULE]
name = "InMemoryModule"  # Built-in for array datasets

[DATAMODULE.dataloader]
batch_size = 64
num_workers = 4

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "data/features.npy"

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"
```

### Config anatomy reference

- `[SESSION]`: `name`, `seed`, `precision`, `root_dir`, `inference` (bool — disables training when true)
- `[MODEL]`: `name`, `module_path`, `checkpoint` (optional path for resuming or inference)
- `[TRAINING]`: `epochs` (top-level shortcut); `[TRAINING.trainer]` = Lightning Trainer kwargs; `[TRAINING.optimizer]` = `name`, `lr`, `weight_decay`; optional `[TRAINING.scheduler]`
- `[DATAMODULE]`: `name`; `[DATAMODULE.dataloader]` = `batch_size`, `num_workers`, etc.
- `[DATASET]`: `name`; `features`/`targets` lists of entries (`name`, `path`, optional `dtype`, `transforms`, `model_input`); `[DATASET.split]` = `val_ratio`, `test_ratio`
- `[MLFLOW]`: `experiment_name`, `run_name`, `tags`, `register_model`, `registered_model_name`, `registered_model_aliases`, `registered_model_version_tags`, `max_retries`
- `[OPTUNA]`: `n_trials`, `direction`, `study_name`, `storage`; optional `[OPTUNA.sampler]` and `[OPTUNA.pruner]`
- `[EXTRAS]`: any free-form keys (ignored by core)

---

## Training

### CLI

```bash
# Basic training
uv run dlkit train config.toml

# Override common parameters without editing the file
uv run dlkit train config.toml --epochs 50 --batch-size 32 --lr 5e-4

# With MLflow tracking (just add [MLFLOW] to your config)
uv run dlkit train config_with_mlflow.toml --experiment-name MyExp --run-name run1
```

### Python API

```python
from dlkit.infrastructure.config import load_settings
from dlkit.interfaces.api import train

cfg = load_settings("config.toml")
result = train(
    cfg,
    epochs=50,
    batch_size=32,
    learning_rate=5e-4,
    experiment_name="MyExp",  # overrides [MLFLOW].experiment_name
    run_name="run1",
)
```

### TrainingResult fields

```python
result.metrics  # dict — val_loss and other logged metrics
result.artifacts  # dict[str, Path] — best_checkpoint, last_checkpoint
result.mlflow_run_id  # str | None
result.mlflow_tracking_uri
result.duration_seconds

# Stacked predictions from the final epoch (TensorDict | None)
stacked = result.stacked
if stacked is not None:
    preds = stacked["predictions"]
    targets = stacked["targets"]
    latents = stacked["latents"]

# Convert to NumPy
arrays = result.to_numpy()
pred_np = result.to_numpy("predictions")
y_np = result.to_numpy(("targets", "y"))  # nested key path
```

### Runtime overrides

These work both as CLI flags and Python keyword arguments:

| Override | CLI flag | Effect |
|----------|----------|--------|
| `epochs` | `--epochs` | `[TRAINING.trainer].max_epochs` |
| `batch_size` | `--batch-size` | `[DATAMODULE.dataloader].batch_size` |
| `learning_rate` | `--lr` | `[TRAINING.optimizer].lr` |
| `experiment_name` | `--experiment-name` | `[MLFLOW].experiment_name` |
| `run_name` | `--run-name` | `[MLFLOW].run_name` |
| `checkpoint_path` | `--checkpoint` | `[MODEL].checkpoint` |
| `output_dir` | `--output-dir` | Output path context |
| `data_dir` | `--data-dir` | Input path context |

---

## Inference

DLKit saves complete model metadata (architecture, hyperparameters, fitted transforms) in
every checkpoint. Inference requires only the checkpoint file — no config file needed.

### Load once, predict many times

```python
from dlkit import load_model
import torch

predictor = load_model(
    "checkpoints/model.ckpt",
    device="auto",  # "auto" | "cpu" | "cuda" | "mps"
    apply_transforms=True,  # Apply saved feature/target transforms (default)
    batch_size=128,  # Optional batch size override
)

# predict() mirrors model.forward() — pass tensors as positional or keyword args
output = predictor.predict(x=torch.randn(32, 10))  # → Tensor
output = predictor.predict(torch.randn(32, 10))  # positional
output = predictor.predict(x=x_t, edge_attr=ea_t)  # multi-input
recon, mu, logvar = predictor.predict(x=x_t)  # multi-output (e.g. VAE)

predictor.unload()  # Free GPU memory when done
```

`apply_transforms=True` (default): raw inputs → feature transforms → model → inverse target
transform → original-space predictions. Pass `apply_transforms=False` when your inputs are
already in the normalized space the model was trained in.

NumPy arrays are not auto-converted; call `torch.from_numpy(arr)` first.

### Context manager (automatic cleanup)

```python
with load_model("model.ckpt", device="cuda") as predictor:
    for batch in my_dataset:
        output = predictor.predict(x=batch)
# GPU memory released automatically
```

### Direct model access

```python
predictor = load_model("model.ckpt")
model = predictor.model  # Standard nn.Module

with torch.no_grad():
    output = model(x)  # Standard forward pass

# Inspect, fine-tune, or export as normal PyTorch
for name, param in model.named_parameters():
    print(name, param.shape)
```

### Checkpoint utilities

```python
from dlkit.interfaces.inference import validate_checkpoint, get_checkpoint_info

info = get_checkpoint_info("model.ckpt")
print(info.model_class, info.shape_summary)

result = validate_checkpoint("model.ckpt")
if result.is_valid:
    print(result.messages)
```

**Checkpoint contract**: checkpoints must contain `dlkit_metadata`; any checkpoint saved by
DLKit's training pipeline satisfies this automatically.

---

## MLflow Tracking

Add an `[MLFLOW]` section to your config to enable experiment tracking. The section's presence
is the enable switch — there is no separate `enabled` flag.

```toml
[MLFLOW]
experiment_name = "my-experiment"
run_name = "baseline"
tags = { team = "research", dataset = "v2" }
register_model = false   # default; set true to register in the Model Registry
```

Infrastructure endpoints belong in environment variables, not TOML:

```bash
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
export MLFLOW_ARTIFACT_URI="s3://my-bucket/mlartifacts"
```

Tracking URI resolution order: `MLFLOW_TRACKING_URI` env → `http://127.0.0.1:5000` if alive → local SQLite fallback.

### Model registry

```python
from dlkit import (
    search_registered_models,
    list_model_versions,
    load_registered_model,
)

tracking_uri = "file:///tmp/mlruns"
model_name = "MyModel"

versions = list_model_versions(model_name, tracking_uri=tracking_uri)

# Load by alias
model = load_registered_model(model_name, alias="production", tracking_uri=tracking_uri)

# Load by version number
model = load_registered_model(model_name, version=versions[-1], tracking_uri=tracking_uri)
```

### Register and annotate from a training result

```python
import dlkit

result = dlkit.train(settings)

version_entity = dlkit.register_logged_model(
    "MyModel",
    run_id=result.mlflow_run_id,
    artifact_path="model",
    tracking_uri=result.mlflow_tracking_uri,
)
version = int(version_entity.version)

dlkit.set_registered_model_alias(
    "MyModel", alias="production", version=version, tracking_uri=result.mlflow_tracking_uri
)
dlkit.set_registered_model_version_tags(
    "MyModel", version=version, tags={"dataset": "v2"}, tracking_uri=result.mlflow_tracking_uri
)
```

Declarative aliases and tags are also supported in TOML:

```toml
[MLFLOW]
register_model = true
registered_model_name = "MyModel"
registered_model_aliases = ["production", "latest"]
registered_model_version_tags = { dataset = "v2" }
```

### Logged-model helpers (when `register_model = false`)

```python
from dlkit import search_logged_models, load_logged_model

results = search_logged_models(
    model_name="MyModel",
    experiment_name="my-experiment",
    tracking_uri="file:///tmp/mlruns",
)
model = load_logged_model(model_uri=results[0].model_uri)
```

---

## Hyperparameter Optimization

Add an `[OPTUNA]` section to run Optuna-based hyperparameter search. Samplable fields are
declared inline with `low`/`high`/`choices` syntax.

### Config

```toml
[SESSION]
name = "hpo_run"

[MODEL]
name = "my_package.models.MyNet"

[MODEL.hidden_size]   # Optuna will sample this field
low = 32
high = 256

[MODEL.num_layers]
choices = [1, 2, 3]

[OPTUNA]
n_trials = 50
direction = "minimize"
study_name = "my_study"
storage = "sqlite:///optuna.db"   # optional persistence

[OPTUNA.sampler]
name = "TPESampler"

[OPTUNA.pruner]
name = "MedianPruner"
n_warmup_steps = 5
```

### CLI

```bash
uv run dlkit optimize config_with_optuna.toml
uv run dlkit optimize config_with_optuna.toml --trials 100 --study-name ablation
```

### Python API

```python
from dlkit.infrastructure.config import load_settings
from dlkit.interfaces.api import optimize

cfg = load_settings("config_with_optuna.toml")
result = optimize(cfg, trials=50, study_name="my_study")

print(result.best_trial)  # Optuna FrozenTrial
print(result.training_result)  # TrainingResult from the best trial
print(result.study_summary)
print(result.duration_seconds)
```

---

## Data Formats

`FlexibleDataset` loads feature and target tensors from:

| Format | Extensions | Notes |
|--------|-----------|-------|
| NumPy single array | `.npy` | Loaded as-is |
| NumPy archive | `.npz` | `name` field selects the array key |
| PyTorch | `.pt`, `.pth` | |
| Text / CSV | `.txt`, `.csv` | |
| Sparse pack | directory | COO format; auto-detected |

### NPZ multi-array archives

The entry `name` is used as the array key inside the `.npz` file:

```toml
[[DATASET.features]]
name = "x"         # loads data.npz["x"]
path = "data.npz"

[[DATASET.features]]
name = "latent"    # loads data.npz["latent"]
path = "data.npz"

[[DATASET.targets]]
name = "y"         # loads data.npz["y"]
path = "data.npz"
```

### Sparse pack context features

For per-sample sparse matrices (e.g., energy-norm loss context matrices), point a feature
entry at a directory containing sparse payload files:

```toml
[[DATASET.features]]
name = "A"
path = "data/matrix_pack"   # directory, not a .npy file
model_input = false          # not passed to model.forward()
loss_input = "A"             # routed to loss function as kwarg
```

Default payload filenames: `indices.npy`, `values.npy`, `nnz_ptr.npy`, `values_scale.npy`
(configurable via `PackFiles`).

Scale contract: `A_original = A_stored × value_scale`. Denormalization applied only when
`SparseFeature(denormalize=True)`.

Sparse pack API:

```python
from dlkit.infrastructure.io import open_sparse_pack, save_sparse_pack, validate_sparse_pack
```

See [`src/dlkit/engine/data/datasets/README.md`](src/dlkit/engine/data/datasets/README.md) for full details.

### In-memory arrays

Pass NumPy arrays directly without creating files:

```python
from dlkit.infrastructure.config import DatasetSettings
from dlkit.infrastructure.config.data_entries import Feature, Target
import numpy as np

dataset_cfg = DatasetSettings(
    name="FlexibleDataset",
    features=(Feature(name="x", value=np.random.randn(1000, 10).astype("float32")),),
    targets=(Target(name="y", value=np.random.randn(1000, 1).astype("float32")),),
)
```

---

## Transforms

Transforms are fitted on training data, persisted in the checkpoint, and automatically
reapplied during inference. This gives scikit-learn style `fit → transform → inverse_transform`
semantics without any glue code.

**Training flow**: Raw Data → forward transform → normalized space → Model → loss (normalized)

**Inference flow**: Raw input → forward transform → Model → inverse transform → original space

### Declaring transforms in config

```toml
[[DATASET.features]]
name = "x"
path = "data/features.npy"
transforms = [
  { name = "MinMaxScaler", module_path = "dlkit.domain.transforms.minmax", dim = 0 },
  { name = "SampleNormL2", module_path = "dlkit.domain.transforms.sample_norm", eps = 1e-8 },
]

[[DATASET.targets]]
name = "y"
path = "data/targets.npy"
transforms = [
  { name = "StandardScaler", module_path = "dlkit.domain.transforms.standard", dim = 0 },
]
```

### Available transforms

| Transform | Module path suffix | Fittable | Invertible | Notes |
|-----------|-------------------|----------|------------|-------|
| `MinMaxScaler` | `transforms.minmax` | yes | yes | Normalizes to [-1, 1] |
| `StandardScaler` | `transforms.standard` | yes | yes | Z-score normalization |
| `PCA` | `transforms.pca` | yes | yes | Dimensionality reduction |
| `SampleNormL2` | `transforms.sample_norm` | no | yes | Per-sample L2 norm |
| `Permutation` | `transforms.permute` | no | yes | Reorder tensor dims |
| `TensorSubset` | `transforms.subset` | yes | yes | Extract sub-tensors |
| `SpectralRadiusNorm` | `transforms.spectral` | no | yes | For matrix inputs |

All module paths are prefixed with `dlkit.domain.`.

### Transform fitting details

- `MinMaxScaler` and `StandardScaler` fit incrementally (batch-by-batch, no full-data buffering)
- `PCA` requires a pre-fitted state (online PCA not yet supported)
- Fitted state is stored in the checkpoint alongside model weights
- Shape information is inferred automatically during `fit()` — no `input_shape` constructor arg needed

### Controlling transforms at inference

```python
# Default: raw input → transforms → model → inverse transforms → original space
predictor = load_model("model.ckpt", apply_transforms=True)
output = predictor.predict(x=raw_data)

# Skip transforms: pass already-normalized data, receive normalized output
predictor = load_model("model.ckpt", apply_transforms=False)
output = predictor.predict(x=normalized_data)

# Manual access to transform chains
from dlkit.interfaces.inference.transforms import load_transforms_from_checkpoint

feature_chains, target_chains = load_transforms_from_checkpoint("model.ckpt")
```

---

## Advanced: Programmatic Configuration

### Hybrid TOML + Python

Load a partial config from TOML and inject sections programmatically. Useful for API
services, parameterized experiments, or workflows where data is generated at runtime.

```toml
# base_config.toml — defines training parameters only
[SESSION]
name = "my_experiment"
seed = 42

[MODEL]
name = "examples.minimal_e2e.model.SimpleNet"

[TRAINING]
[TRAINING.trainer]
max_epochs = 100
[TRAINING.optimizer]
name = "Adam"
lr = 1e-3
```

```python
from dlkit.infrastructure.config import DatasetSettings, load_settings
from dlkit.infrastructure.config.data_entries import Feature, Target
from dlkit.interfaces.api import train

cfg = load_settings("base_config.toml")

# Inject DATASET section
cfg = cfg.patch({
    "DATASET": DatasetSettings(
        name="FlexibleDataset",
        features=(Feature(name="x", path="/data/features.npy"),),
        targets=(Target(name="y", path="/data/labels.npy"),),
    )
})

result = train(cfg, epochs=50)
```

### In-memory arrays (zero file I/O)

```python
import numpy as np

features = np.random.randn(1000, 10).astype("float32")
labels = np.random.randn(1000, 1).astype("float32")

cfg = cfg.patch({
    "DATASET": DatasetSettings(
        name="FlexibleDataset",
        features=(Feature(name="x", value=features),),
        targets=(Target(name="y", value=labels),),
    )
})

result = train(cfg, epochs=10)
```

### Experiment loops

```python
from pathlib import Path

datasets = [
    ("exp1", Path("/data/exp1")),
    ("exp2", Path("/data/exp2")),
]

base_cfg = load_settings("base_config.toml")

for name, data_dir in datasets:
    cfg = base_cfg.patch({
        "SESSION.name": name,
        "DATASET": DatasetSettings(
            name="FlexibleDataset",
            features=(Feature(name="x", path=str(data_dir / "features.npy")),),
            targets=(Target(name="y", path=str(data_dir / "labels.npy")),),
        ),
    })
    result = train(cfg)
    print(f"{name}: {result.metrics}")
```

### Updating nested settings

`update_settings()` deep-merges changes into an existing settings object and returns a new
validated instance:

```python
from dlkit.infrastructure.config.core.updater import update_settings

cfg = update_settings(
    cfg,
    {
        "TRAINING": {"optimizer": {"lr": 5e-4, "weight_decay": 1e-5}},
        "DATAMODULE": {"dataloader": {"batch_size": 128}},
    },
)
```

---

## Loss Pairing

- **Strict mapping**: predictions are paired to targets by name
- **Single-target fallback**: if exactly one target and one prediction exist, they are paired even if names differ
- **Autoencoders**: set `[WRAPPER].is_autoencoder = true`; features are used as targets automatically when no explicit targets are defined
- **Multi-target**: return a `dict` from `model.forward()` with keys matching target names; single tensor return is fine for single-target models

### Advanced loss keyword arguments

Pass additional context tensors (e.g., sparse matrices) to the loss function:

```toml
[[DATASET.features]]
name = "A"
path = "data/context.npy"
model_input = false    # not passed to model.forward()
loss_input = "A"       # passed to loss(predictions, target, A=A)
```

See [`src/dlkit/engine/adapters/lightning/README.md`](src/dlkit/engine/adapters/lightning/README.md) and
[`src/dlkit/engine/training/execution.md`](src/dlkit/engine/training/execution.md) for
full loss pairing and metric routing documentation.

---

## Custom Components

Register your own models, datasets, losses, metrics, and datamodules by name and reference
them in configs without fully qualified paths.

```python
from dlkit import register_model, register_dataset, register_loss, register_metric


@register_model(name="MyNet", aliases=["my_net"])
class MyNet(torch.nn.Module): ...


@register_dataset(name="ToyDataset")
class ToyDataset(torch.utils.data.Dataset): ...


@register_loss(name="mae", aliases=["l1"])
class MAELoss(torch.nn.Module): ...
```

Then reference by name in config:

```toml
[MODEL]
name = "MyNet"

[DATASET]
name = "ToyDataset"
```

Use `use=True` to force selection without a config name:

```python
@register_model(use=True)
class MyNet(torch.nn.Module): ...
```

Resolution order: `use=True` override → registered name/alias → dotted import path.

---

## Environment and Paths

### Root directory resolution

DLKit resolves the project root in this order (highest to lowest priority):

1. `DLKIT_ROOT_DIR` environment variable
2. `[SESSION].root_dir` in the loaded config
3. Current working directory

Standard subdirectories (`output/`, `checkpoints/`, `mlruns/`, `mlartifacts/`, `splits/`) are
computed relative to the root via `dlkit.infrastructure.io.locations`:

```python
from dlkit.infrastructure.io import locations

print(locations.output())
print(locations.checkpoints_dir())
print(locations.mlruns_dir())
```

### Optional PATHS section

```toml
[PATHS]
data_dir    = "./data"
output_dir  = "./results"
checkpoint_path = "./checkpoints/model.ckpt"
```

All values are resolved relative to `SESSION.root_dir` via DLKit's `SecurePath` system.

### Logging

| Variable | Effect |
|----------|--------|
| `DLKIT_LOG_LEVEL` | Log level (default: `INFO`) |
| `DLKIT_LOG_BACKTRACE=1` | Extended call-chain display in exceptions |
| `DLKIT_LOG_DIAGNOSE=1` | Local-variable dump in tracebacks (verbose) |

`DLKIT_LOG_DIAGNOSE=1` can flood output with full tensor reprs when an exception is
captured — use only when debugging.

---

## Contributing

Bug reports and contributions are welcome at <https://github.com/constatza/dlkit/issues>.

### Contributor Setup

For local development, install [uv](https://docs.astral.sh/uv/) and
[prek](https://github.com/j178/prek) first, then sync the pinned dev toolchain and
install the git hooks:

```bash
uv sync --dev
uv tool install prek
prek install -t pre-commit -t pre-push
```

Run repository-local developer tools through `uv run` so you use the pinned
toolchain from this project:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check src tests
uv run pytest tests -m "not slow" --maxfail=1 --disable-warnings -q
uv run pytest tests --maxfail=1 --disable-warnings -vv
```
