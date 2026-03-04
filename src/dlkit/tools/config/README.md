# DLKit Configuration System

Pydantic-based typed configuration for DLKit workflows.

## Overview

DLKit uses a hierarchical TOML configuration system with Pydantic validation. The configuration is organized into top-level sections that map directly to Pydantic settings classes.

## Settings Hierarchy

```
GeneralSettings
├── SESSION          → SessionSettings
├── MODEL            → ModelComponentSettings
├── TRAINING         → TrainingSettings
│   ├── trainer      → TrainerSettings
│   ├── optimizer    → OptimizerSettings
│   └── scheduler    → SchedulerSettings
├── DATAMODULE       → DataModuleSettings
│   └── dataloader   → DataloaderSettings
├── DATASET          → DatasetSettings
│   ├── features[]   → Feature/PathFeature
│   ├── targets[]    → Target/PathTarget
│   └── split        → IndexSplitSettings
├── MLFLOW           → MLflowSettings
├── OPTUNA           → OptunaSettings
│   ├── sampler      → SamplerSettings
│   └── pruner       → PrunerSettings
├── PATHS            → PathsSettings
└── EXTRAS           → ExtrasSettings
```

---

## SessionSettings

Top-level session configuration controlling execution mode and global settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"dlkit-session"` | Session name for identification |
| `inference` | `bool` | `False` | Run in inference mode when true |
| `seed` | `int` | `1` | Random seed for reproducibility |
| `precision` | `PrecisionStrategy` | `FULL_32` | Computation precision strategy |
| `root_dir` | `SecurePath \| None` | `None` | Root directory for path resolution |

### Precision Strategy Reference

| Enum Value | Lightning Param | torch.dtype | Aliases |
|------------|-----------------|-------------|---------|
| `FULL_64` | `64` | `float64` | `double`, `float64`, `f64`, `fp64` |
| `FULL_32` | `32` | `float32` | `single`, `float32`, `f32`, `fp32` |
| `MIXED_16` | `"16-mixed"` | `float16` | `mixed16`, `mixed_16` |
| `TRUE_16` | `"16-true"` | `float16` | `half`, `float16`, `f16`, `fp16` |
| `MIXED_BF16` | `"bf16-mixed"` | `bfloat16` | `bfloat16_mixed`, `bf16_mixed` |
| `TRUE_BF16` | `"bf16-true"` | `bfloat16` | `bfloat16`, `bf16` |

---

## ModelComponentSettings

Model architecture configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str \| type[nn.Module]` | required | Model class name or type |
| `module_path` | `str` | `"dlkit.core.models.nn"` | Module path to the model |
| `checkpoint` | `Path \| None` | `None` | Checkpoint path for inference (weights only) |

### Model Hyperparameters

| Field | Type | Description |
|-------|------|-------------|
| `heads` | `int \| None` | Number of attention heads |
| `num_layers` | `int \| None` | Number of layers |
| `hidden_size` | `int \| None` | Hidden dimension size |
| `latent_size` | `int \| None` | Latent dimension size |
| `kernel_size` | `int \| None` | Convolution kernel size |
| `in_channels` | `int \| None` | Number of input channels |
| `out_channels` | `int \| None` | Number of output channels |

---

## TrainingSettings

Core training configuration with nested library-specific settings.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | `int` | `100` | Number of training epochs |
| `patience` | `int` | `10` | Early stopping patience |
| `monitor_metric` | `str` | `"val_loss"` | Metric to monitor for early stopping |
| `mode` | `str` | `"min"` | Monitoring mode (`min` or `max`) |
| `resume_from_checkpoint` | `Path \| None` | `None` | Checkpoint for resuming training (full state) |
| `trainer` | `TrainerSettings` | defaults | PyTorch Lightning trainer settings |
| `optimizer` | `OptimizerSettings` | defaults | Optimizer configuration |
| `scheduler` | `SchedulerSettings \| None` | `None` | Learning rate scheduler |
| `loss_function` | `LossComponentSettings` | defaults | Loss function configuration |
| `metrics` | `tuple[MetricComponentSettings]` | `()` | Metrics to compute |

### TrainerSettings

PyTorch Lightning Trainer configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_epochs` | `int` | `100` | Maximum training epochs |
| `gradient_clip_val` | `float \| None` | `None` | Gradient clipping value |
| `fast_dev_run` | `bool \| int` | `False` | Fast development run |
| `accelerator` | `str` | `"auto"` | Hardware accelerator (`cpu`, `gpu`, `auto`, `tpu`) |
| `enable_checkpointing` | `bool` | `False` | Enable automatic checkpointing |
| `precision` | `str \| int \| None` | `None` | Override session precision |
| `callbacks` | `tuple[CallbackSettings]` | `()` | Lightning callbacks |
| `logger` | `LoggerSettings` | defaults | Logger configuration |

### OptimizerSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"AdamW"` | Optimizer name |
| `module_path` | `str` | `"torch.optim"` | Module path |
| `lr` | `float` | `1e-3` | Learning rate (alias: `learning_rate`) |
| `weight_decay` | `float` | `0.0` | Weight decay |

### SchedulerSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"ReduceLROnPlateau"` | Scheduler name |
| `module_path` | `str` | `"torch.optim.lr_scheduler"` | Module path |
| `factor` | `float` | `0.5` | Reduction factor |
| `patience` | `int` | `1000` | Patience before reducing LR |
| `min_lr` | `float` | `1e-8` | Minimum learning rate |

---

## DataModuleSettings

Data loading configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"InMemoryDataModule"` | DataModule class name |
| `module_path` | `str` | `"dlkit.core.datamodules"` | Module path |
| `dataloader` | `DataloaderSettings` | defaults | DataLoader configuration |

### DataloaderSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | `int` | `64` | Batch size |
| `num_workers` | `int` | `cpu_count - 1` | Number of worker processes |
| `shuffle` | `bool` | `True` | Shuffle training data |
| `persistent_workers` | `bool` | `True` | Keep workers alive |
| `pin_memory` | `bool` | `True` | Pin memory for GPU transfer |

---

## DatasetSettings

Dataset-specific configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"FlexibleDataset"` | Dataset class name |
| `module_path` | `str` | `"dlkit.core.datasets"` | Module path |
| `type` | `DatasetFamily \| None` | `None` | Dataset family hint |
| `root` | `DirectoryPath \| None` | `None` | Root directory (alias: `root_dir`) |
| `features` | `list[FeatureType]` | `[]` | Feature entries |
| `targets` | `list[TargetType]` | `[]` | Target entries |
| `split` | `IndexSplitSettings` | defaults | Train/val/test split |

### Feature/Target Configuration

Features and targets support path-based (loaded from disk) or value-based (in-memory) configurations.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Entry name |
| `path` | `Path` | required | Path to data file |
| `dtype` | `torch.dtype \| None` | `None` | Override data type |
| `required_in_loss` | `bool` | features: `False`, targets: `True` | Include in loss |
| `transforms` | `list[TransformSettings]` | `[]` | Transform chain |
| `write` | `bool` (targets only) | `False` | Save predictions |

### IndexSplitSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `filepath` | `FilePath \| None` | `None` | Existing split file |
| `test_ratio` | `float` | `0.15` | Test set fraction (alias: `test`) |
| `val_ratio` | `float` | `0.15` | Validation fraction (alias: `val`) |

---

## MLflowSettings

MLflow experiment tracking configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable MLflow tracking |
| `experiment_name` | `str` | `"Experiment"` | Experiment name |
| `run_name` | `str \| None` | `None` | Run name |
| `register_model` | `bool` | `True` | Register trained models |
| `registered_model_name` | `str \| None` | `None` | Optional registered model name override |
| `registered_model_aliases` | `tuple[str, ...] \| None` | `None` | Optional aliases applied to each registered version |
| `registered_model_version_tags` | `dict[str, str] \| None` | `None` | Optional model-version tags applied after registration |
| `max_trials` | `int` | `3` | Max connection attempts |

Infrastructure URIs are env-driven and not accepted in TOML:
- `MLFLOW_TRACKING_URI`
- `MLFLOW_ARTIFACT_URI`

---

## OptunaSettings

Hyperparameter optimization configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable Optuna optimization |
| `n_trials` | `int` | `3` | Number of optimization trials |
| `direction` | `str` | `"minimize"` | Optimization direction |
| `study_name` | `str \| None` | `None` | Study name for persistence |
| `storage` | `str \| None` | `None` | Storage URL (e.g., `sqlite:///optuna.db`) |
| `sampler` | `SamplerSettings` | defaults | Sampler configuration |
| `pruner` | `PrunerSettings` | defaults | Pruner configuration |
| `model` | `dict` | `{}` | Model parameter ranges |

### SamplerSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"TPESampler"` | Sampler algorithm |
| `module_path` | `str` | `"optuna.samplers"` | Module path |
| `seed` | `int \| None` | `None` | Random seed |

### PrunerSettings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"NopPruner"` | Pruner algorithm |
| `module_path` | `str` | `"optuna.pruners"` | Module path |
| `n_warmup_steps` | `int \| None` | `None` | Warmup steps |
| `interval_steps` | `int \| None` | `None` | Pruning interval |

---

## PathsSettings

Optional standardized paths with automatic resolution.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_dir` | `SecurePath \| None` | `None` | Output directory |
| `checkpoint_path` | `SecurePath \| None` | `None` | Checkpoint file |
| `data_dir` | `SecurePath \| None` | `None` | Datasets directory |
| `weights_path` | `SecurePath \| None` | `None` | Model weights file |
| `config_path` | `SecurePath \| None` | `None` | Additional config files |

Extras allowed for custom user-defined paths.

---

## ExtrasSettings

Free-form user-defined settings container.

- Accepts any keys/values (no predefined fields)
- Not used by core DLKit logic
- Useful for custom scripts and tools

---

## Loading Configuration

### From TOML File

```python
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml_file("config.toml")
```

### Programmatic Configuration

```python
from dlkit.tools.config import (
    GeneralSettings,
    SessionSettings,
    ModelComponentSettings,
    TrainingSettings,
    DatasetSettings,
)
from dlkit.tools.config.data_entries import Feature, Target

settings = GeneralSettings(
    SESSION=SessionSettings(
        name="my-experiment",
        seed=42,
        precision="float32",
    ),
    MODEL=ModelComponentSettings(
        name="ConstantWidthFFNN",
        hidden_size=256,
        num_layers=4,
    ),
    TRAINING=TrainingSettings(
        epochs=100,
        patience=10,
    ),
    DATASET=DatasetSettings(
        features=[Feature(name="x", path="data/features.npy")],
        targets=[Target(name="y", path="data/targets.npy")],
    ),
)
```

### Section-Specific Loading

```python
from dlkit.tools.config.factories import load_sections

sections = load_sections("config.toml", ["SESSION", "MODEL", "TRAINING"])
```

---

## Path Resolution

DLKit uses a three-layer path resolution hierarchy:

1. **Thread-local context** (`PathOverrideContext`): Set by CLI/API during execution
2. **DLKitEnvironment fallback**: From `DLKIT_ROOT_DIR` env var or `SESSION.root_dir`
3. **CWD fallback**: Current working directory as last resort

Standard locations computed relative to root:
- `output/` - Output files
- `checkpoints/` - Model checkpoints
- `mlruns/` - MLflow runs
- `mlartifacts/` - MLflow artifacts
- `splits/` - Data splits
