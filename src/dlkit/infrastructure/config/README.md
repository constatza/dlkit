# DLKit Configuration System

DLKit uses typed Pydantic settings plus runtime-owned module defaults.

## Key Points
- `infrastructure.io` owns TOML loading and section discovery.
- `infrastructure.config` owns typed settings, validation, and patching.
- Workflow settings are split across `workflow_settings_base.py`, `training_workflow_settings.py`, and `inference_workflow_settings.py`.
- `workflow_settings.py` remains the re-export shim.
- The public config surface uses the canonical workflow classes directly; legacy alias exports were removed.
- Secure URI/path config types live under `infrastructure.config.security.uri_types`.
- `DATASET.family` can explicitly select the runtime dataset family.
- `infrastructure.precision` owns the precision service — see [`../precision/README.md`](../precision/README.md).

## Recommended Entry Points
```python
from dlkit.infrastructure.config.factories import load_settings
from dlkit.infrastructure.config.workflow_configs import (
    TrainingWorkflowConfig,
    OptimizationWorkflowConfig,
    InferenceWorkflowConfig,
)

# Load typed workflow config — type depends on SESSION.workflow in the TOML
settings = load_settings("config.toml")  # -> TrainingWorkflowConfig | OptimizationWorkflowConfig | InferenceWorkflowConfig

# Load partial sections only
from dlkit.infrastructure.config.factories import load_sections_config
partial = load_sections_config("config.toml", ["MODEL", "DATASET"])
```

---

## Optimization Settings

`TRAINING.optimizer` holds an `OptimizerPolicySettings` object. The simplest
case needs no explicit TOML — defaults to AdamW at `lr=1e-3`.

### Choosing an optimizer

`default_optimizer` is a discriminated union keyed on `name`. Supported values:
`"AdamW"`, `"Adam"`, `"LBFGS"`, `"Muon"`, `"Concurrent"`.

**AdamW (default — can be omitted entirely)**
```toml
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 0.01
```

**Adam**
```toml
[TRAINING.optimizer.default_optimizer]
name = "Adam"
lr = 5e-4
weight_decay = 0.0
```

**Muon (requires `muon` extra)**
```toml
[TRAINING.optimizer.default_optimizer]
name = "Muon"
lr = 0.02
momentum = 0.95
ns_steps = 5
```

**LBFGS**
```toml
[TRAINING.optimizer.default_optimizer]
name = "LBFGS"
lr = 1.0
max_iter = 20
```

### Adding a learning-rate scheduler

`default_scheduler` is also a discriminated union keyed on `name`. Supported
values: `"ReduceLROnPlateau"`, `"StepLR"`, `"CosineAnnealingLR"`,
`"CosineAnnealingWarmRestarts"`.

```toml
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[TRAINING.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
min_lr = 1e-6
```

```toml
[TRAINING.optimizer.default_scheduler]
name = "CosineAnnealingLR"
T_max = 100
eta_min = 1e-6
```

### Concurrent optimizers (Muon + AdamW)

`ConcurrentOptimizerSettings` (name `"Concurrent"`) fits anywhere an optimizer
fits — use it as `default_optimizer` for the common case, or inside a stage for
concurrent-then-sequential programs.

Omitting `selectors` is only valid when at least one sub-optimizer is `"Muon"`
(auto-infers Muon-eligible vs. non-Muon partitioning). For any other split,
provide explicit `selectors` — one per optimizer.

**TOML — Muon + AdamW with auto-inferred selectors:**
```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
```

**TOML — encoder/decoder split with explicit selectors:**
```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Adam", lr = 1e-3}, {name = "Adam", lr = 5e-4}]
selectors  = [{prefix = "encoder"},        {prefix = "decoder"}]
```

**Python:**
```python
from dlkit.infrastructure.config.optimizer_component import (
    ConcurrentOptimizerSettings, MuonSettings, AdamWSettings,
)

OptimizerPolicySettings(
    default_optimizer=ConcurrentOptimizerSettings(
        optimizers=(MuonSettings(lr=0.02), AdamWSettings(lr=3e-4))
    )
)
```

### Multi-stage optimizer programs

For sequential stage switching (e.g. warm-up then fine-tune), populate
`TRAINING.optimizer.stages`. See
[`../../engine/training/optimization/README.md`](../../engine/training/optimization/README.md)
for full examples.

### Python API

```python
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamWSettings,
    AdamSettings,
    MuonSettings,
    ReduceLROnPlateauSettings,
    CosineAnnealingLRSettings,
)
from dlkit.infrastructure.config.training_settings import TrainingSettings

# Default: AdamW lr=1e-3, no scheduler
settings = TrainingSettings()

# Custom optimizer
settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=AdamSettings(lr=5e-4, weight_decay=1e-4),
    )
)

# Optimizer + scheduler
settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=AdamWSettings(lr=1e-3, weight_decay=0.01),
        default_scheduler=ReduceLROnPlateauSettings(
            mode="min", patience=10, factor=0.5
        ),
    )
)

# Muon optimizer
settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=MuonSettings(lr=0.02, momentum=0.95, ns_steps=5),
    )
)
```

Patching an existing config programmatically:

```python
from dlkit.infrastructure.config import update_settings

# Update only the learning rate, preserving all other fields
new_settings = update_settings(
    settings,
    {"TRAINING": {"optimizer": {"default_optimizer": {"lr": 5e-4}}}},
)
```
