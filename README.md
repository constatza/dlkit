# DLKit

[![Tests](https://github.com/constatza/dlkit/actions/workflows/testing.yml/badge.svg)](https://github.com/constatza/dlkit/actions/workflows/testing.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue?style=flat-square)](#installation)

DLKit is a typed deep-learning workflow toolkit for training, optimization, and checkpoint-based inference on top of PyTorch and Lightning.

[Installation](#installation) • [Quick Start](#quick-start) • [CLI Commands](#cli-commands) • [Configuration Model](#configuration-model) • [Training](#training) • [Optimization](#optimization) • [Inference](#inference) • [Python API](#python-api)

## Features

- Typed TOML-first workflows for training, optimization, and inference.
- Programmatic APIs for running the same workflows from Python.
- MLflow integration for run tracking and model registration.
- Optuna integration for hyperparameter search.
- Entry-based dataset configuration with explicit feature and target routing.
- Support for staged and concurrent optimizer policies (including Muon).
- Multirun sweeps and sample-size convergence studies as first-class workflows.

## Installation

DLKit currently targets Python `>=3.14,<3.15`.

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first if you do not already use it.

PyTorch is selected through extras and is not installed by default. Choose exactly one accelerator extra:

- `cu130` for CUDA 13.0
- `cu128` for CUDA 12.8
- `cpu` for CPU-only installs

### Add To A Project

Use this when you want `import dlkit` inside an application or library.

```bash
uv add "dlkit[cu130] @ git+https://github.com/constatza/dlkit.git"
```

Replace `cu130` with `cu128` or `cpu` if you need a different build.

### Install The CLI As A Tool

Use this when you only want the `dlkit` command for config-driven workflows.

```bash
uv tool install "dlkit[cu130] @ git+https://github.com/constatza/dlkit.git"
```

Replace `cu130` with `cu128` or `cpu` if you need a different build.

## Quick Start

Generate a training template, edit it, then validate it:

```bash
uv run dlkit config create --output train.toml --type training
uv run dlkit config validate train.toml
```

For inference:

```bash
uv run dlkit config create --output inference.toml --type inference
uv run dlkit predict inference.toml path/to/model.ckpt
```

If you installed the CLI with `uv tool install`, drop the `uv run` prefix.

## CLI Commands

| Command | Purpose |
| --- | --- |
| `dlkit train CONFIG.toml` | Train a model |
| `dlkit predict CONFIG.toml CHECKPOINT` | Batch prediction from a checkpoint |
| `dlkit evaluate CONFIG.toml CHECKPOINT` | Stats/plots for a checkpoint against a labeled split |
| `dlkit evaluate-multirun CONFIG.toml` | Batch-evaluate every child run of a sweep |
| `dlkit optimize CONFIG.toml --trials N` | Run an Optuna hyperparameter search |
| `dlkit optimize status \| plot STUDY STORAGE` | Inspect or plot an Optuna study |
| `dlkit converge CONFIG.toml` | Sample-size convergence study |
| `dlkit multirun run \| validate CONFIG.toml` | Execute, or dry-run, a batch sweep of child configs |
| `dlkit convert CHECKPOINT OUTPUT` | Export a checkpoint to ONNX |
| `dlkit config validate \| show \| create \| sync-templates` | Config validation, inspection, and template generation |

Run `dlkit --help` or `dlkit <command> --help` for full options. See the [CLI command reference](src/dlkit/interfaces/cli/commands/commands.md) for details.

## Configuration Model

DLKit uses `run.type` to select the runtime path:

- `train`
- `predict` (inference)
- `search` (hyperparameter optimization)
- `convergence` (sample-size convergence studies)
- `multirun` (batch sweeps of child configs)
- `fit` (one-shot, non-gradient model fits)

The dataset model is entry-based. Features and targets are declared with `[[data.features]]` and `[[data.targets]]` blocks instead of a single shorthand dataset path.

By default, DLKit maps named model-input features to `model.forward()` by keyword. If `x` and `z` are declared as named features, DLKit calls `model(x=x_tensor, z=z_tensor)`. Unnamed model-input features use positional dispatch.

### Minimal Training Config

```toml
[run]
type = "train"
seed = 42
precision = "32"

[experiment]
name = "my_training_session"

[model]
class = "your.model.class"

[data]
class = "FlexibleDataset"

[[data.features]]
name = "x"
path = "features.npy"

[[data.targets]]
name = "y"
path = "targets.npy"

[data.splits]
val = 0.15
test = 0.15

[training]
loss = "mse"

[training.trainer]
max_epochs = 100
accelerator = "auto"

[training.optimizer]
name = "AdamW"
lr = 1e-3
```

### Entry Routing Example

Use `model_input`, `loss_input`, and `write` when you need more than a plain feature or target:

```toml
[[data.features]]
name = "stiffness"
path = "stiffness.npy"
model_input = false
loss_input = "K"

[[data.targets]]
name = "prediction"
path = "targets.npy"
write = true
```

`model_input = false` keeps an entry out of `model.forward()`. `loss_input = "K"` routes it into the loss function as a named kwarg. `write = true` marks an entry for prediction/latent writing during inference workflows.

An entry's `data_role` (`feature`/`target`/`latent`/`auxiliary`) is inferred from which list it's declared in — `[[data.features]]` entries are always `feature`, `[[data.targets]]` entries are always `target` — so it never needs to be set explicitly here.

Default `forward()` mapping rules:

- Named features with `model_input = true` are passed by keyword.
- The entry `name` must match the corresponding `model.forward()` parameter name.
- Unnamed features with `model_input = true` use positional dispatch.
- Features with `model_input = false` are excluded from `model.forward()`.
- `loss_input` affects loss-function kwargs only; it does not change model dispatch.

Keyword-dispatch example:

```toml
[[data.features]]
name = "x"
path = "features_x.npy"

[[data.features]]
name = "z"
path = "features_z.npy"
```

```python
def forward(self, x, z):
    ...
```

DLKit dispatches these as `model(x=x_tensor, z=z_tensor)`.

Legacy positional example:

```python
from dlkit.infrastructure.config.data_entries import ValueEntry
from dlkit.infrastructure.config.data_roles import DataRole

features = [
    ValueEntry(name=None, value=x_array, data_role=DataRole.FEATURE),
    ValueEntry(name=None, value=z_array, data_role=DataRole.FEATURE),
]
```

```python
def forward(self, x, z):
    ...
```

Because these model-input entries are unnamed, DLKit uses positional dispatch and calls `model(x_tensor, z_tensor)`.

## Training

For config-driven training:

```bash
uv run dlkit train train.toml
uv run dlkit train train.toml --epochs 10 --batch-size 32 --learning-rate 5e-4
uv run dlkit train train.toml --checkpoint path/to/last.ckpt
```

For programmatic training:

```python
from dlkit import train
from dlkit.interfaces.api.domain import TrainingOverrides
from dlkit.settings import load_job

settings = load_job("train.toml")

result = train(
    settings,
    overrides=TrainingOverrides(
        epochs=10,
        batch_size=32,
        learning_rate=5e-4,
    ),
)

print(result.metrics)
print(result.checkpoint_path)
```

## Optimization

Optimization is a separate workflow selected with `run.type = "search"` and a `[search]` section.

`[search.space]` defines the hyperparameter search space. Each entry is keyed by a dotted config path (e.g. `model.hidden_size`, `training.optimizer.lr`) and a typed range object: `float`, `log_float`, `int`, `log_int`, or `categorical`.

```toml
[run]
type = "search"
seed = 42
precision = "32"

[experiment]
name = "search_run"

[model]
class = "your.model.class"

[search]
n_trials = 50
study_name = "baseline_search"
storage = "sqlite:///optuna.db"

[search.space]
"model.hidden_size" = { type = "categorical", choices = [64, 128, 256] }
"model.num_layers" = { type = "categorical", choices = [2, 4, 6] }
"training.optimizer.lr" = { type = "log_float", low = 1e-4, high = 1e-2 }

[training]
loss = "mse"

[training.trainer]
max_epochs = 25
accelerator = "auto"

[data]
class = "FlexibleDataset"

[[data.features]]
name = "x"
path = "features.npy"

[[data.targets]]
name = "y"
path = "targets.npy"
```

Run it from the CLI:

```bash
uv run dlkit optimize optimize.toml --trials 50 --study-name baseline_search
```

Or from Python:

```python
from dlkit import optimize
from dlkit.interfaces.api.domain import OptimizationOverrides
from dlkit.settings import load_job

settings = load_job("optimize.toml")

result = optimize(
    settings,
    overrides=OptimizationOverrides(
        trials=50,
        study_name="baseline_search",
    ),
)

print(result.best_trial)
```

## Inference

### Config-Driven Batch Inference

Inference configs use `run.type = "predict"` and `model.checkpoint`:

```toml
[run]
type = "predict"
seed = 42
precision = "32"

[model]
class = "your.model.class"
checkpoint = "./model.ckpt"

[data]
class = "FlexibleDataset"

[[data.features]]
name = "x"
path = "features.npy"
```

Current CLI behavior still takes an explicit checkpoint argument, so use:

```bash
uv run dlkit predict inference.toml path/to/model.ckpt
```

### Direct Python Inference

```python
from dlkit import load_model

with load_model("path/to/model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

## Python API

The top-level package exposes a curated workflow surface:

- `train`
- `evaluate`
- `optimize`
- `execute`
- `load_model`
- `load_training_config`
- `load_inference_config`
- `load_optimization_config`
- `register_model`
- `register_dataset`

Typical usage:

```python
from dlkit import load_model, train
from dlkit.interfaces.api.domain import TrainingOverrides
from dlkit.settings import load_job

settings = load_job("train.toml")
result = train(settings, overrides=TrainingOverrides(epochs=10))

with load_model(result.checkpoint_path, device="auto") as predictor:
    output = predictor.predict(x=batch)
```

### Top-Level Shim Modules

Convenience re-export modules for common import paths, alongside the curated surface above:

- `dlkit.nn` / `dlkit.gnn` — model and layer families
- `dlkit.settings` — `load_job` and typed config models
- `dlkit.errors` — the DLKit exception hierarchy
- `dlkit.results` — workflow result value objects
- `dlkit.io` — file I/O and path resolution helpers
- `dlkit.config` — `load_training_config` / `load_inference_config` / `load_optimization_config`
- `dlkit.inference` — checkpoint loading and prediction
- `dlkit.mlflow` — MLflow tracking and model-registry helpers
- `dlkit.registry` — `register_model` / `register_dataset`

## More Reference

- [Configuration module](src/dlkit/infrastructure/config/config.md)
- [Optimizer policy reference](src/dlkit/engine/training/optimization/optimization.md)
- [CLI command reference](src/dlkit/interfaces/cli/commands/commands.md)
- [Integration testing notes](tests/integration/README.md)
