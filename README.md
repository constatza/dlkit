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

## Inference (New)

DLKit now provides a dedicated inference system that operates independently from training configurations. **No configuration files are required** - everything needed is extracted from the model checkpoint using enhanced checkpoint metadata that includes model settings, shape specifications, and transform configurations.

### Default Method: Direct Inference

The **recommended and default approach** requires only a checkpoint and input data:

```python
from dlkit.interfaces.api import infer

# Primary/default inference method - no config files needed
# Automatic model reconstruction from enhanced checkpoint metadata
result = infer(
    checkpoint_path="model.ckpt",  # Contains all necessary metadata
    inputs={"x": torch.randn(32, 10)},
    batch_size=16,
    apply_transforms=True  # Automatically applies saved transforms
)
predictions = result.predictions
```

### Alternative Configuration Methods

**File-based inputs:**
```python
# Direct file input
result = infer(
    checkpoint_path="model.ckpt",
    inputs={"features": "data/test_features.csv"},
    batch_size=32
)
```

**Manual InferenceConfig (optional):**
```python
from dlkit.interfaces.inference import InferenceConfig
from dlkit.interfaces.inference.api import infer_with_config

config = InferenceConfig(
    model_checkpoint_path="model.ckpt",
    batch_size=16,
    device="cuda",
    apply_transforms=True
)
result = infer_with_config(config, inputs)
```

**Lightning-based prediction (uses training config):**
```python
from dlkit.interfaces.api import predict_with_config
from dlkit.tools.config import load_settings

# For scenarios requiring Lightning framework and training datasets
cfg = load_settings("config.toml", inference=True)
result = predict_with_config(cfg, "model.ckpt")
```

## Breaking Changes

### Inference API Changes
- **`infer()` function**: Now dedicated to inference only (lightweight, checkpoint-based)
- **`predict_with_config()` function**: New function for Lightning-based simple prediction using training configs
- **`InferenceWorkflowSettings`**: Removed from config system
- **`load_inference_settings()`**: ❌ **REMOVED** - no longer exists
- **Default inference method**: Now direct `infer(checkpoint_path, inputs)` - no settings loading required
- **Configuration files**: Not needed for basic inference - everything extracted from checkpoint

### Config System Updates
- **Optional PATHS section**: New `[PATHS]` section available for standardized path configuration
- **Partial config loading**: New efficient section-based loading APIs in `dlkit.tools.io.config`
- **Path resolution**: Enhanced SecurePath system with automatic resolution
- **Enhanced checkpoint format**: All checkpoints now include comprehensive metadata for shape-free inference

### Migration Guide
```python
# ❌ OLD (no longer works)
from dlkit.interfaces.api import infer
from dlkit.tools.config import load_inference_settings  # REMOVED
cfg = load_inference_settings("config.toml")  # REMOVED
result = infer(cfg)  # Old signature

# ✅ NEW - Default/Recommended Method
from dlkit.interfaces.api import infer
# No config loading needed - direct inference
result = infer(
    checkpoint_path="model.ckpt",
    inputs=your_input_data,  # tensors, dicts, arrays, or file paths
    batch_size=32,  # optional
    device="auto"   # optional
)

# ✅ Alternative: Lightning-based prediction (if you need training config)
from dlkit.interfaces.api import predict_with_config
from dlkit.tools.config import load_settings
cfg = load_settings("config.toml", inference=True)
result = predict_with_config(cfg, "model.ckpt")
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

DLKit automatically applies transforms during inference using configurations saved in the enhanced checkpoint format. This includes applying direct transforms to feature tensors before model invocation and applying inverse transforms to predictions/targets in the predict output. **No configuration files are needed** - transforms are automatically restored from checkpoint metadata.

- Configuration: attach transforms to `[DATASET].features` and `[DATASET].targets` entries. Example (excerpt):

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

- Programmatic toggles on the wrapper (default True):
  - `model.apply_feature_transforms`: when False, direct transforms are not applied before inference.
  - `model.apply_inverse_target_transforms`: when False, inverse transforms are not applied to predictions/targets in the predict output.

Example:
```python
# Given a built wrapper and datamodule
model.apply_feature_transforms = True
model.apply_inverse_target_transforms = True
preds = trainer.predict(model, datamodule=dm)
```

- Manual helpers (optional):
  - `model.feature_transforms({"x": x_raw})` → dict with transformed features
  - `model.target_transforms_inverse({"y": y_pred})` → dict with inverse-transformed predictions

Notes
- Transform chains are automatically saved in the enhanced checkpoint metadata and restored during inference. DLKit fits chains once on training data and reuses them for validation/test/predict.
- Enhanced checkpoints include complete transform state, enabling shape-free inference without manual configuration.
- For graph/timeseries wrappers that manage their own featurization/encoders, entry-based transforms are ignored by design to avoid conflicts.

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
