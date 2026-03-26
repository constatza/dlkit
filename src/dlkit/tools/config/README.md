# DLKit Configuration System

DLKit uses typed Pydantic settings plus protocol-oriented TOML loading.

## Public API Layers

There are two supported ways to work with config:

1. Workflow loaders for application code
2. Full settings models and low-level section readers for advanced tooling

### Workflow loaders

Use these for normal training workflows:

```python
from dlkit.tools.config import load_settings, load_sections

settings = load_settings("config.toml")  # TrainingWorkflowSettings
partial = load_sections("config.toml", ["MODEL", "DATASET"])
```

- `load_settings()` returns `TrainingWorkflowSettings`
- `load_sections()` returns `BaseWorkflowSettings` with only the requested sections populated

### Full-model access

Use `GeneralSettings` when you want the full top-level schema as a single model:

```python
from dlkit.tools.config import GeneralSettings

settings = GeneralSettings.from_toml_file("config.toml")
```

`GeneralSettings` is still the canonical aggregate model, but the high-level loader API is workflow-oriented.

## Settings Hierarchy

```text
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
│   ├── features[]   → PathFeature / SparseFeature / ValueFeature
│   ├── targets[]    → PathTarget / ValueTarget
│   └── split        → IndexSplitSettings
├── MLFLOW           → MLflowSettings
├── OPTUNA           → OptunaSettings
│   ├── sampler      → SamplerSettings
│   └── pruner       → PrunerSettings
├── PATHS            → PathsSettings
└── EXTRAS           → ExtrasSettings
```

Workflow-specialized models reuse the same sections:

```text
TrainingWorkflowSettings
├── SESSION
├── MODEL
├── DATAMODULE
├── DATASET
├── TRAINING
├── MLFLOW
├── OPTUNA
├── PATHS
└── EXTRAS
```

## MLflowSettings

`MLflowSettings` is a flat, client-only model. The presence of an `[MLFLOW]`
section enables tracking; there is no separate `enabled` flag.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiment_name` | `str` | `"Experiment"` | Experiment name |
| `run_name` | `str \| None` | `None` | Optional run name |
| `tags` | `dict[str, str] \| None` | `None` | Tags attached to every run |
| `register_model` | `bool` | `False` | Register logged model artifacts |
| `registered_model_name` | `str \| None` | `None` | Optional registered model name override |
| `registered_model_aliases` | `tuple[str, ...] \| None` | `None` | Optional aliases applied after registration |
| `registered_model_version_tags` | `dict[str, str] \| None` | `None` | Optional model-version tags applied after registration |
| `max_retries` | `int` | `3` | Max retry attempts for transient client operations |

Infrastructure stays in environment variables, not TOML:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_ARTIFACT_URI`

Fail-fast validation rejects removed or env-only TOML keys:

- Legacy nested sections: `server`, `client`
- Removed enablement field: `enabled`
- Env-only infra fields: `tracking_uri`, `artifacts_destination`

Example:

```toml
[MLFLOW]
experiment_name = "baseline"
run_name = "ffnn-01"
tags = { team = "platform", dataset = "A" }
register_model = false
registered_model_name = "FFNN"
registered_model_aliases = ["candidate", "dataset_A_latest"]
registered_model_version_tags = { dataset = "A", team = "platform" }
max_retries = 5
```

## Config Protocols

Low-level config behavior is described by protocols in `dlkit.tools.io.protocols`:

- `ConfigParser`: parse full config or named sections
- `SectionExtractor`: extract top-level sections from parsed config data
- `ConfigValidator[T]`: validate sections into Pydantic models
- `PartialConfigReader`: high-level section-reading contract

These protocols describe the public design boundary. The current implementation uses:

- `DLKitTomlSource` in `dlkit.tools.config.core.sources`
- Section mapping helpers in `dlkit.tools.io.config`
- Strict override patching in `dlkit.tools.config.core.patching`

## Low-Level Section Loading

The low-level readers support registry-based or explicit model loading:

```python
from dlkit.tools.config import SessionSettings
from dlkit.tools.io.config import (
    get_available_sections,
    load_section_config,
    load_sections_config,
)

sections = load_sections_config("config.toml", ["SESSION", "TRAINING"])
session = load_section_config("config.toml", SessionSettings)
model = load_section_config("config.toml", section_name="MODEL")
available = get_available_sections("config.toml")
```

Useful helpers from `dlkit.tools.io.config`:

- `register_section_mapping(model_class, section_name)`
- `reset_section_mappings(section_name=None)`
- `get_section_name(model_class)`
- `get_model_class_for_section(section_name)`

## Environment Override Protocol

Workflow settings apply overrides in this order:

1. TOML file
2. Environment patches
3. Explicit runtime overrides

Environment overrides use `DLKIT_<SECTION>__<field>`:

```bash
export DLKIT_SESSION__precision=double
export DLKIT_MLFLOW__experiment_name=benchmark
export DLKIT_TRAINING__epochs=200
```

Overrides are merged with strict validated patching. Colliding dotted/nested patches raise errors instead of silently overwriting values.

## Path Resolution

`DLKitTomlSource` preprocesses paths before validation:

- `SESSION.root_dir`
- `TRAINING.trainer.default_root_dir`
- `MODEL.checkpoint`
- `DATASET.split.filepath`
- `DATASET.features[*].path`
- `DATASET.targets[*].path`
- `PATHS.*`

Resolution order is:

1. Active path override context
2. `SESSION.root_dir`
3. Config file directory / current working directory fallback

## Immutable Settings

All `BasicSettings` subclasses are `frozen=True`. Fields cannot be mutated after construction.
Use `patch_model()` from `dlkit.tools.config.core.patching` for structured updates:

```python
from dlkit.tools.config.core.patching import patch_model

new_settings = patch_model(settings, {"TRAINING.epochs": 200, "SESSION.seed": 0})
```

`patch_model()` returns a **new** instance — the original is unchanged. Colliding dotted-key patches raise `ValueError` to prevent silent overwrites.

## Programmatic Construction

```python
from dlkit.tools.config import (
    GeneralSettings,
    SessionSettings,
    ModelComponentSettings,
    TrainingSettings,
    DatasetSettings,
    MLflowSettings,
)
from dlkit.tools.config.data_entries import Feature, Target

settings = GeneralSettings(
    SESSION=SessionSettings(name="my-experiment", seed=42, precision="float32"),
    MODEL=ModelComponentSettings(name="ConstantWidthFFNN", hidden_size=256, num_layers=4),
    TRAINING=TrainingSettings(epochs=100, patience=10),
    DATASET=DatasetSettings(
        features=[Feature(name="x", path="data/features.npy")],
        targets=[Target(name="y", path="data/targets.npy")],
    ),
    MLFLOW=MLflowSettings(experiment_name="baseline"),
)
```
