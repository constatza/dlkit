# DLKit

[![Tests](https://github.com/constatza/dlkit/actions/workflows/testing.yml/badge.svg)](https://github.com/constatza/dlkit/actions/workflows/testing.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue?style=flat-square)](#installation)

DLKit is a typed deep-learning workflow toolkit for training, optimization, and checkpoint-based inference on top of PyTorch and Lightning.

[Installation](#installation) • [Quick Start](#quick-start) • [Configuration Model](#configuration-model) • [Training](#training) • [Optimization](#optimization) • [Inference](#inference) • [Python API](#python-api)

## Features

- Typed TOML-first workflows for training, optimization, and inference.
- Programmatic APIs for running the same workflows from Python.
- MLflow integration for run tracking and model registration.
- Optuna integration for hyperparameter search.
- Entry-based dataset configuration with explicit feature and target routing.
- Support for staged and concurrent optimizer policies.

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

## Configuration Model

DLKit uses `SESSION.workflow` to select the runtime path:

- `train`
- `optimize`
- `inference`

The current dataset model is entry-based. Features and targets are declared with `[[DATASET.features]]` and `[[DATASET.targets]]` blocks instead of a single shorthand dataset path.

By default, DLKit maps model-input features to `model.forward()` positionally, in `[[DATASET.features]]` config order. If `x` is declared before `z`, DLKit calls `model(x_tensor, z_tensor)`, not `model(x=x_tensor, z=z_tensor)`.

### Minimal Training Config

```toml
[SESSION]
name = "my_training_session"
workflow = "train"
seed = 42
precision = "32"
root_dir = "./"

[MODEL]
name = "your.model.class"

[TRAINING.trainer]
max_epochs = 100
accelerator = "auto"

[DATAMODULE]
name = "your.datamodule.class"

[DATASET]
name = "FlexibleDataset"
root_dir = "./data"

[[DATASET.features]]
name = "x"
path = "features.npy"
data_role = "feature"
field_role = "feature"

[[DATASET.targets]]
name = "y"
path = "targets.npy"
data_role = "target"
field_role = "target"
```

### Entry Routing Example

Use `model_input`, `loss_input`, and `write` when you need more than a plain feature or target:

```toml
[[DATASET.features]]
name = "stiffness"
path = "stiffness.npy"
data_role = "feature"
model_input = false
loss_input = "K"

[[DATASET.features]]
name = "query_coords"
path = "query_coords.npy"
data_role = "feature"
field_role = "target_coordinates"

[[DATASET.targets]]
name = "prediction"
path = "targets.npy"
data_role = "target"
write = true
```

`model_input = false` keeps an entry out of `model.forward()`. `loss_input = "K"` routes it into the loss function as a named kwarg. `write = true` marks an entry for prediction/latent writing during inference workflows.

Default `forward()` mapping rules:

- Features with `model_input = true` are passed positionally.
- Positional order matches `[[DATASET.features]]` order in the config.
- Features with `model_input = false` are excluded from `model.forward()`.
- `loss_input` affects loss-function kwargs only; it does not change model dispatch.

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
from dlkit.settings import load_settings

settings = load_settings("train.toml")

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

Optimization is a separate workflow selected with `SESSION.workflow = "optimize"` and an `[OPTUNA]` section.

```toml
[SESSION]
name = "search_run"
workflow = "optimize"
seed = 42
precision = "32"
root_dir = "./"

[MODEL]
name = "your.model.class"

[OPTUNA]
enabled = true
n_trials = 50
study_name = "baseline_search"
storage = "sqlite:///optuna.db"

[TRAINING.trainer]
max_epochs = 25
accelerator = "auto"

[DATAMODULE]
name = "your.datamodule.class"

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "features.npy"
data_role = "feature"

[[DATASET.targets]]
name = "y"
path = "targets.npy"
data_role = "target"
```

Run it from the CLI:

```bash
uv run dlkit optimize optimize.toml --trials 50 --study-name baseline_search
```

Or from Python:

```python
from dlkit import optimize
from dlkit.interfaces.api.domain import OptimizationOverrides
from dlkit.settings import load_settings

settings = load_settings("optimize.toml")

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

Inference configs use `SESSION.workflow = "inference"` and `MODEL.checkpoint`:

```toml
[SESSION]
name = "my_inference_session"
workflow = "inference"
seed = 42
precision = "32"
root_dir = "./"

[MODEL]
name = "your.model.class"
checkpoint = "./model.ckpt"

[DATASET]
name = "FlexibleDataset"

[[DATASET.features]]
name = "x"
path = "features.npy"
data_role = "feature"
field_role = "feature"
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
from dlkit.settings import load_settings

settings = load_settings("train.toml")
result = train(settings, overrides=TrainingOverrides(epochs=10))

with load_model(result.checkpoint_path, device="auto") as predictor:
    output = predictor.predict(x=batch)
```

## More Reference

- [Configuration module](src/dlkit/infrastructure/config/config.md)
- [Optimizer policy reference](src/dlkit/engine/training/optimization/optimization.md)
- [CLI command reference](src/dlkit/interfaces/cli/commands/commands.md)
- [Integration testing notes](tests/integration/README.md)
