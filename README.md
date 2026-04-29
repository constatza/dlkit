# DLKit

DLKit is a typed, configuration-driven training and inference toolkit built on
PyTorch and Lightning. The repository currently exposes training, optimization,
checkpoint-based inference, config/template utilities, and MLflow model helpers.

## Current Surface

The current public surface is:

- top-level `dlkit`: `train`, `optimize`, `load_model`, `validate_config`,
  MLflow model helpers, registry helpers, and config I/O helpers
- `dlkit.settings`: `load_settings`, `load_sections`, `GeneralSettings`,
  `TrainingWorkflowSettings`, and related typed settings models
- `dlkit.interfaces.api`: `train`, `optimize`, and `execute`

Important current-state notes:

- `execute()` is available from `dlkit.interfaces.api`, not from top-level `dlkit`
- `load_settings()` currently returns `TrainingWorkflowSettings`
- the repository does not currently ship a checked-in top-level `examples/` directory
- CLI prediction is currently exposed as `dlkit predict entry ...`; for the most
  stable inference workflow, prefer Python `load_model()`

## Installation

DLKit requires Python `3.14.x`.

Install dependencies with an explicit Torch extra:

```bash
uv sync --extra cpu
```

GPU extras are also available:

```bash
uv sync --extra cu128
uv sync --extra cu130
```

The project does not define a default Torch backend; one of `cpu`, `cu128`, or
`cu130` must be selected.

## Quick Start

Generate a training template:

```bash
uv run dlkit config create --output config.toml --type training
```

Train from the CLI:

```bash
uv run dlkit train config.toml
uv run dlkit train config.toml --epochs 5 --batch-size 32 --learning-rate 5e-4
```

Train from Python:

```python
from dlkit import train
from dlkit.settings import load_settings

settings = load_settings("config.toml")
result = train(
    settings,
    overrides={
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 5e-4,
    },
)

print(result.metrics)
print(result.checkpoint_path)
```

Run checkpoint inference from Python:

```python
from dlkit import load_model
import torch

with load_model("model.ckpt", device="auto") as predictor:
    output = predictor.predict(x=torch.randn(8, 10))
```

Current CLI prediction entrypoint:

```bash
uv run dlkit predict entry config.toml model.ckpt --batch-size 64
```

## Configuration

DLKit ships template generators for:

- `training`
- `inference`
- `mlflow`
- `optuna`

Example:

```bash
uv run dlkit config create --output optuna.toml --type optuna
```

The generated training template currently looks like this:

```toml
[SESSION]
name = "my_session"
inference = false
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

[EXTRAS]
example_key = "user_defined_value"
```

## Settings Model Reality

The current settings story is not yet fully unified:

- `dlkit.settings.load_settings("config.toml")` returns `TrainingWorkflowSettings`
- runtime workflow APIs accept
  `GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig`
- runtime coercion converts workflow config variants into `GeneralSettings`
  before execution
- `GeneralSettings.from_toml_file("config.toml")` is the current flattened-model
  loader

That means the repository exposes both a flattened settings model and
workflow-specific settings models today.

## API and CLI Entry Points

Primary current entry points:

- CLI training: `dlkit train CONFIG.toml`
- CLI optimization: `dlkit optimize CONFIG.toml`
- Python training: `from dlkit import train`
- Python optimization: `from dlkit import optimize`
- Unified Python execution: `from dlkit.interfaces.api import execute`
- Checkpoint inference: `from dlkit import load_model`

Common CLI training overrides:

| Override | Flag |
|---|---|
| epochs | `--epochs` |
| batch size | `--batch-size` |
| learning rate | `--learning-rate` |
| checkpoint | `--checkpoint` |
| root directory | `--root-dir` |
| output directory | `--output-dir` |
| dataflow directory | `--dataflow-dir` |
| MLflow experiment name | `--experiment-name` |
| MLflow run name | `--run-name` |

Programmatic overrides are passed as `overrides={...}` dictionaries, not as
direct keyword override arguments.

## MLflow and Optuna

Tracking and optimization are configured by adding `[MLFLOW]` or `[OPTUNA]`
sections to the config. Current behavior:

- MLflow tracking is activated by presence of the `[MLFLOW]` section
- Optuna optimization requires `[OPTUNA].enabled = true`
- CLI optimization uses `dlkit optimize ...`
- unified Python execution can route to optimization when `OPTUNA.enabled` is true

These capabilities are configurable at runtime, but their packages are currently
installed as default dependencies rather than optional extras.

## Validation

You can validate a loaded settings object from Python:

```python
from dlkit import validate_config
from dlkit.settings import load_settings

settings = load_settings("config.toml")
validate_config(settings)
```

CLI validation:

```bash
uv run dlkit config validate config.toml
```

## Architecture

The repository is organized around this package DAG:

```text
interfaces -> engine, domain, infrastructure, common
engine -> domain, infrastructure, common
domain -> common
infrastructure -> common, infrastructure.precision
common -> (none)
infrastructure.precision -> (none)
```

## Related Docs

- `docs/architecture/README.md`
- `docs/external_ux_architecture_review.md`
- `src/dlkit/interfaces/api/api.md`
- `src/dlkit/interfaces/inference/README.md`
