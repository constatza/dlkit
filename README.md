# DLKit

[![Tests](https://github.com/constatza/dlkit/actions/workflows/testing.yml/badge.svg)](https://github.com/constatza/dlkit/actions/workflows/testing.yml)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue?style=flat-square)](#installation)

Typed, configuration-driven training, optimization, and checkpoint inference on top of PyTorch and Lightning.

[Installation](#installation) • [Quick Start](#quick-start) • [Configuration](#configuration) • [Training](#training) • [Optimization](#optimization) • [Outputs and MLflow](#outputs-and-mlflow) • [Inference](#inference) • [Python API](#python-api)

## Features

DLKit is built for projects that want reproducible ML workflows without giving up programmatic control.

- Train models from TOML configs or Python.
- Run Optuna studies from the same configuration surface.
- Track runs and model artifacts with MLflow.
- Load checkpoints for direct Python inference or config-driven batch prediction.
- Use simple single-optimizer training, sequential multi-stage optimizer programs, or concurrent optimizers on disjoint parameter sets.
- Keep workflow configuration typed and immutable, with explicit patching instead of hidden mutation.

## Installation

DLKit currently targets Python `>=3.14,<3.15`.

PyTorch is selected through extras and is **not** installed by default. Choose exactly one of:
- `cpu`
- `cu128`
- `cu130`

### Add To A Project

Use this when you want the Python API (`import dlkit`) inside your project.

For CPU:

```bash
uv add "dlkit[cpu] @ git+ssh://git@github.com/constatza/dlkit.git"
```

For CUDA 12.8:

```bash
uv add "dlkit[cu128] @ git+ssh://git@github.com/constatza/dlkit.git"
```

For CUDA 13.0:

```bash
uv add "dlkit[cu130] @ git+ssh://git@github.com/constatza/dlkit.git"
```

### Install The CLI As A Tool

Use this when you only want the `dlkit` command for config-driven workflows.

For CPU:

```bash
uv tool install "dlkit[cpu] @ git+ssh://git@github.com/constatza/dlkit.git"
```

For CUDA 12.8:

```bash
uv tool install "dlkit[cu128] @ git+ssh://git@github.com/constatza/dlkit.git"
```

For CUDA 13.0:

```bash
uv tool install "dlkit[cu130] @ git+ssh://git@github.com/constatza/dlkit.git"
```

## Quick Start

The shortest verified path today is:

```bash
uv run dlkit config create --output train.toml --type training
uv run dlkit config validate train.toml

uv run dlkit config create --output inference.toml --type inference
uv run dlkit predict inference.toml path/to/model.ckpt
```

If you installed the CLI as a tool instead, use `dlkit ...` directly.

```python
from dlkit import train
from dlkit.settings import load_settings

settings = load_settings("train.toml")
result = train(settings)

print(result.checkpoint_path)
print(result.metrics)
```

DLKit is TOML-first for workflow execution. `SESSION.workflow` selects the runtime path:
- `train`
- `optimize`
- `inference`

## Minimal Training Config

A minimal training config follows the current generated template shape:

```toml
[SESSION]
name = "my_session"
workflow = "train"
seed = 42
precision = "medium"

[MODEL]
name = "your.model.class"

[TRAINING.trainer]
max_epochs = 100
accelerator = "auto"

[DATAMODULE]
name = "your.datamodule.class"

[DATASET]
name = "your.dataset.class"
```

For a larger example, see [docs/examples/full_config_example.toml](docs/examples/full_config_example.toml).

## Configuration

DLKit supports two ways of working with settings.

### 1. Config-Driven Workflows

This is the recommended default for reproducible training, optimization, and inference.

- Use `dlkit config create` to scaffold a starting point.
- Keep workflow selection in `SESSION.workflow`.
- Use `[MLFLOW]` when you want run tracking.
- Use `[OPTUNA]` when you want hyperparameter optimization.
- Use `MODEL.checkpoint` for inference configs, not for resuming training.

Currently exposed template families:
- `training`
- `inference`
- `mlflow`
- `optuna`

For optimization templates, verify that the generated config uses `SESSION.workflow = "optimize"` before running it.

### 2. Programmatic Workflows

Use Python when you want typed overrides, environment-aware orchestration, or application-specific setup around the workflow.

- `dlkit.settings.load_settings("config.toml")` loads the correct workflow config based on `SESSION.workflow`.
- Settings objects are immutable; use `.patch(...)` when you want to derive a modified config.
- `dlkit.config` exposes workflow-specific loaders if you prefer concrete entry points.

## Training

The most reliable training path today is the Python API:

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
```

A `dlkit train` CLI command also exists, but the runtime override layer is currently typed around `TrainingOverrides`. For now, prefer the Python API when you want the verified training path.

Current training overrides supported by `TrainingOverrides` include:
- `epochs`
- `batch_size`
- `learning_rate`
- `checkpoint_path`
- `root_dir`
- `output_dir`
- `data_dir`
- `experiment_name`
- `run_name`
- `register_model`
- `tags`
- `loss_function`
- `loss_module`

The CLI surface currently exposes these flags:

```bash
uv run dlkit train train.toml
uv run dlkit train train.toml --epochs 10 --batch-size 32 --learning-rate 5e-4
uv run dlkit train train.toml --checkpoint path/to/last.ckpt
```

The current CLI surface exposes these flags:
- `--epochs`
- `--batch-size`
- `--learning-rate`
- `--checkpoint`
- `--root-dir`
- `--output-dir`
- `--experiment-name`
- `--run-name`
- `--mlflow`

### Two-Stage Optimizer Programs

DLKit supports sequential optimizer stages through `TRAINING.optimizer.stages`.

```toml
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
trigger = {at_epoch = 10}

[[TRAINING.optimizer.stages]]
optimizer = {name = "LBFGS", lr = 1.0}
```

Schedulers follow the optimizer-policy shape:

- use `TRAINING.optimizer.default_scheduler` for single-stage programs with no
  explicit `stages`
- use `TRAINING.optimizer.stages[*].scheduler` for staged programs

```toml
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
```

This is useful for schedules such as:
- warm up with AdamW, then refine with LBFGS
- coarse stage, then low-learning-rate fine tuning
- plateau-triggered stage switching

DLKit also supports concurrent optimizers on disjoint parameter sets through
`name = "Concurrent"`. For deeper staged and concurrent examples, see
[src/dlkit/engine/training/optimization/optimization.md](src/dlkit/engine/training/optimization/optimization.md).

### Training Resume vs Inference Checkpoints

These are separate concepts in the current config model:

- Use `TRAINING.resume_from_checkpoint` to resume training state.
- Use `MODEL.checkpoint` for inference workflows.

## Optimization

Optimization is a distinct workflow, not just a flag on training.

A verified optimization config needs:
- `SESSION.workflow = "optimize"`
- an `[OPTUNA]` section
- `OPTUNA.enabled = true`
- an `[OPTUNA.model]` section with search ranges

Example:

```toml
[SESSION]
name = "search_run"
workflow = "optimize"
seed = 42
precision = "medium"

[MODEL]
name = "your.model.class"
hidden_size = 128

[OPTUNA]
enabled = true
n_trials = 50
direction = "minimize"
study_name = "baseline_search"

[OPTUNA.model]
hidden_size = [64, 512]
num_layers = [2, 8]
```

Run it from Python with the typed optimization API:

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
print(result.study_summary)
```

Notes:
- `OPTUNA.enabled = true` is required by the optimization workflow.
- `OPTUNA.model` should mirror tunable fields from `MODEL`.
- `study_name` controls persistent study naming when storage is configured.
- A `dlkit optimize` CLI command exists, but for the verified optimization path today, prefer the Python API with `OptimizationOverrides`.
- The CLI also exposes `dlkit optimize status ...` and `dlkit optimize plot ...` for existing studies.

## Outputs and MLflow

### Local Workflow Results

In Python, training returns a `TrainingResult` that exposes:
- `metrics`
- `artifacts`
- `checkpoint_path`
- `predictions`
- `mlflow_run_id`
- `mlflow_tracking_uri`

`checkpoint_path` resolves to the best checkpoint when available, and otherwise falls back to the last checkpoint if one was collected from Lightning checkpoint callbacks.

### MLflow Tracking

MLflow is enabled by the **presence** of an `[MLFLOW]` section. There is no `enabled` flag.

```toml
[MLFLOW]
experiment_name = "my_experiment"
run_name = "baseline"
register_model = true
registered_model_name = "my_model"
```

Infrastructure endpoints are environment-driven:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_ARTIFACT_URI=file:///absolute/path/to/mlruns
```

You can also force tracking from Python or the CLI:
- Python: `train(..., mlflow=True)` or `optimize(..., mlflow=True)`
- CLI surfaces also expose `--mlflow` on training and optimization commands

For advanced logged-model and registry workflows, use the `dlkit.mlflow` namespace:
- `search_logged_models`
- `load_logged_model`
- `register_logged_model`
- `search_registered_models`
- `load_registered_model`

## Inference

DLKit exposes two verified inference paths.

### Config-Driven Batch Inference

```bash
uv run dlkit predict inference.toml path/to/model.ckpt
```

Current CLI behavior requires an explicit checkpoint argument. The inference config may also include `MODEL.checkpoint`, but the CLI command still takes a concrete checkpoint path today.

### Direct Python Inference

```python
from dlkit import load_model

with load_model("path/to/model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=batch)
    predictions = output.predictions
```

Additional user-facing inference helpers are available under `dlkit.inference`:
- `load_model`
- `load_model_from_settings`
- `validate_checkpoint`
- `get_checkpoint_info`

## Python API

A minimal typed Python workflow looks like this:

```python
from dlkit import load_model, train
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
print(result.mlflow_run_id)

with load_model(result.checkpoint_path, device="auto") as predictor:
    output = predictor.predict(x=batch)
```

Useful public namespaces:
- `dlkit`: curated top-level workflow and inference surface
- `dlkit.config`: user-facing configuration types and loaders
- `dlkit.settings`: settings models plus `load_settings()`
- `dlkit.mlflow`: MLflow run-artifact and model-registry helpers
- `dlkit.registry`: decorators and registry introspection for custom components

## More Documentation

For deeper reference material:
- [Configuration reference](src/dlkit/infrastructure/config/config.md)
- [Optimizer policy reference](src/dlkit/engine/training/optimization/optimization.md)
- [Architecture overview](docs/architecture/README.md)
- [Testing notes](tests/TESTING.md)
