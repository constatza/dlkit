# Configuration Module

`dlkit.infrastructure.config` owns typed settings, validation, patching,
workflow-specific config views, and component-setting models.

## Responsibilities

- immutable Pydantic settings models
- workflow settings and workflow-specific config views
- patch application and runtime override support
- component settings and factory support
- security-oriented URI and path config types
- load-time validation for importable component `module_path` values

Precision is documented in [`../precision/precision.md`](../precision/precision.md).

## Current Structure

- `core/`: base settings, patching, factories, build context
- `model_components.py`: model, loss, metric, and wrapper component settings
- `workflow_settings_base.py`, `training_workflow_settings.py`, `inference_workflow_settings.py`: workflow-specific settings
- `workflow_settings.py`: re-export shim for workflow settings
- `dataset_settings.py`: dataset config, including explicit `family`
- `optimization_trigger.py`: `TriggerSettings`
- `optimization_selector.py`: `ParameterSelectorSettings`
- `optimization_stage.py`: `OptimizationStageSettings`
- `optimizer_policy.py`: `OptimizerPolicySettings`
- `optimizer_component.py`: concrete optimizer and scheduler component settings
- `security/uri_types.py`: secure URI and path config types

## Optimization Settings

`TRAINING.optimizer` holds an `OptimizerPolicySettings` object.

- When `stages` is empty, use `default_optimizer` and optional
  `default_scheduler`.
- When `stages` is populated, each `OptimizationStageSettings` defines its own
  optimizer, optional scheduler, optional selector, and optional trigger.
- Scheduler runtime semantics live in
  [`../../engine/training/optimization/optimization.md`](../../engine/training/optimization/optimization.md).

### Choosing an optimizer

`default_optimizer` accepts the built-in optimizer settings:
- `"AdamW"`
- `"Adam"`
- `"LBFGS"`
- `"Muon"`
- `"Concurrent"`

```toml
[TRAINING.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 0.01
```

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
```

### Adding a scheduler

`default_scheduler` and `stages[*].scheduler` accept the built-in scheduler
settings:
- `"ReduceLROnPlateau"`
- `"StepLR"`
- `"CosineAnnealingLR"`
- `"CosineAnnealingWarmRestarts"`

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
[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
scheduler = {name = "StepLR", step_size = 10, gamma = 0.5}
trigger = {at_epoch = 10}

[[TRAINING.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

### Concurrent optimizers

`ConcurrentOptimizerSettings` fits anywhere an optimizer fits.

- Omit `selectors` only when at least one sub-optimizer is `"Muon"`.
- For all other concurrent splits, provide one selector per optimizer.

```toml
[TRAINING.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Adam", lr = 1e-3}, {name = "Adam", lr = 5e-4}]
selectors  = [{prefix = "encoder"}, {prefix = "decoder"}]
```

### Python API

```python
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    ConcurrentOptimizerSettings,
    MuonSettings,
    ReduceLROnPlateauSettings,
)
from dlkit.infrastructure.config.training_settings import TrainingSettings

settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=AdamWSettings(lr=1e-3, weight_decay=0.01),
        default_scheduler=ReduceLROnPlateauSettings(
            mode="min",
            patience=10,
            factor=0.5,
        ),
    )
)

settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=ConcurrentOptimizerSettings(
            optimizers=(MuonSettings(lr=0.02), AdamSettings(lr=3e-4))
        )
    )
)
```

Patching an existing config programmatically:

```python
from dlkit.infrastructure.config import update_settings

new_settings = update_settings(
    settings,
    {"TRAINING": {"optimizer": {"default_optimizer": {"lr": 5e-4}}}},
)
```

## Ownership Boundary

- `infrastructure.io` reads TOML files and resolves sections.
- `infrastructure.config` validates those payloads into typed settings models
  and applies runtime overrides.

## `PATHS` Contract

`PATHS` is a path-only settings section.

- Declared fields such as `output_dir` and `data_dir` use the `SecurePath`
  contract.
- Extra keys are allowed for user-defined path names, but they are normalized
  with the same `SecurePath` rules as declared fields.
- Canonical runtime representation is a normalized POSIX-style string.
- Non-path arbitrary values do not belong in `PATHS`; put them in `EXTRAS`.
- `SESSION.root_dir` remains the only root-setting authority. `PATHS` values are
  resolved relative to the existing path-resolution precedence, but `PATHS`
  does not define an alternate project root.

## Notes

- `DATASET.family` is the explicit dataset-family override. Runtime heuristics
  only apply when it is unset.
- Component `module_path` values remain optional; when provided they are
  validated at config load time, and runtime builders still apply default module
  namespaces when omitted.
- `InferenceWorkflowConfig.has_dataset_config` is the explicit predicate for
  dataset-backed batch prediction.
