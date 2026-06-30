# Configuration Module

`dlkit.infrastructure.config` owns typed settings, validation, patching,
workflow-specific config views, and component-setting models.

## Responsibilities

- immutable Pydantic settings models (`frozen=True`)
- `JobConfig` top-level discriminated union (training / inference / search)
- TOML loading via `load_job()` with deep-merge and profile references
- patch application and runtime override support
- component settings and factory support
- security-oriented URI and path config types
- load-time validation for importable component `module_path` values

Precision is documented in [`../precision/precision.md`](../precision/precision.md).

## Current Structure

- `core/`: base settings, patching, factories, build context, TOML source
- `core/_path_helpers.py`: path-preprocessing helpers (training / model / data)
- `job_config.py`: `JobConfig`, `TrainingJobConfig`, `InferenceJobConfig`, `SearchJobConfig`
- `run_settings.py`: `RunSettings` (type, seed, precision)
- `experiment_settings.py`: `ExperimentSettings` (name, run_name, register_model)
- `model_components.py`: canonical `ModelComponentSettings`, plus loss/metric component settings
- `data_settings.py`: `DataSettings` plus entry types
- `training_settings.py`: `TrainingSettings`, `StoppingSettings`
- `search_settings.py`: `SearchSettings`, param types
- `tracking_settings.py`: `TrackingSettings`
- `optimizer_policy.py`: `OptimizerPolicySettings`
- `optimizer_component.py`: concrete optimizer and scheduler component settings

## Loading a Config

```python
from dlkit.infrastructure.config.factories import load_job

job = load_job("config.toml")                # type inferred from run.type
job = load_job(["base.toml", "local.toml"]) # merged left-to-right
job = load_job("config.toml", run_type="train")  # override type
```

## Optimization Settings

`training.optimizer` holds an `OptimizerPolicySettings` object.

`search.space` defines hyperparameter search ranges.
Each entry's `choices` must contain only scalar persistable values:
`None`, `bool`, `int`, `float`, or `str`. Structured categorical choices such
as lists are rejected during config validation instead of being forwarded to
Optuna with persistence warnings.

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
- `"BatchedMuon"`
- `"Concurrent"`

```toml
[training.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3
weight_decay = 0.01
```

```toml
[training.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Muon", lr = 0.02}, {name = "AdamW", lr = 3e-4}]
```

```toml
[training.optimizer.default_optimizer]
name = "BatchedMuon"
lr = 0.02
```

### Adding a scheduler

`default_scheduler` and `stages[*].scheduler` accept the built-in scheduler
settings:
- `"ReduceLROnPlateau"`
- `"StepLR"`
- `"CosineAnnealingLR"`
- `"CosineAnnealingWarmRestarts"`

```toml
[training.optimizer.default_optimizer]
name = "AdamW"
lr = 1e-3

[training.optimizer.default_scheduler]
name = "ReduceLROnPlateau"
mode = "min"
factor = 0.5
patience = 10
min_lr = 1e-6
```

```toml
[[training.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-3}
scheduler = {name = "StepLR", step_size = 10, gamma = 0.5}
trigger = {at_epoch = 10}

[[training.optimizer.stages]]
optimizer = {name = "AdamW", lr = 1e-4}
```

### Muon learning-rate defaults

`MuonSettings` and `BatchedMuonSettings` default `adjust_lr_fn` to
`"match_rms_adamw"`. This follows the PyTorch Muon mode intended for reusing
AdamW-tuned learning rate and weight decay values.

DLKit supports two Muon configuration modes:

- Convenience mode: a lone `MuonSettings` / `BatchedMuonSettings` auto-splits
  into Muon-family plus internal companion AdamW.
- Explicit mode: `ConcurrentOptimizerSettings(optimizers=(MuonSettings(...), AdamWSettings(...)))`
  gives independent control over Muon and companion AdamW settings.

In convenience mode, DLKit keeps one configured `lr` for both the Muon-family
side and the companion AdamW side. The Muon-family side applies its own
RMS-matching adjustment internally; the companion AdamW side uses the configured
`lr` directly.

### Concurrent optimizers

`ConcurrentOptimizerSettings` fits anywhere an optimizer fits.

- Omit `selectors` only when **exactly one** sub-optimizer is `"Muon"` or `"BatchedMuon"`.
  The builder assigns `MuonEligibleSelector` to that single Muon-family optimizer and
  `NonMuonSelector` to the rest. Having two Muon-family optimizers with no selectors
  raises `ValidationError` because both would receive the same parameters.
- For all other concurrent splits, provide one selector per optimizer.

```toml
[training.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [{name = "Adam", lr = 1e-3}, {name = "Adam", lr = 5e-4}]
selectors  = [{prefix = "encoder"}, {prefix = "decoder"}]
```

```toml
[training.optimizer.default_optimizer]
name = "Concurrent"
optimizers = [
  {name = "Muon", lr = 0.02, adjust_lr_fn = "match_rms_adamw"},
  {name = "AdamW", lr = 3e-4, weight_decay = 0.01},
]
```

### Python API

```python
from dlkit.infrastructure.config.optimizer_policy import OptimizerPolicySettings
from dlkit.infrastructure.config.optimizer_component import (
    AdamSettings,
    AdamWSettings,
    BatchedMuonSettings,
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

settings = TrainingSettings(
    optimizer=OptimizerPolicySettings(
        default_optimizer=BatchedMuonSettings(lr=0.02)
    )
)
```

## Ownership Boundary

- `infrastructure.io` reads TOML files and resolves sections.
- `infrastructure.config` validates those payloads into typed settings models
  and applies runtime overrides.

## Path Ownership

- Relative config paths resolve from the config file location during TOML
  preprocessing (`core/_path_helpers.py`).
- `data.root` is the root-like path anchor for dataset entry paths and split
  file paths.
- Output ownership stays with the producing subsystem:
  `training.trainer.default_root_dir` for Lightning-local work when MLflow is
  disabled, and MLflow artifact/storage URIs when tracking is enabled.

## Notes

- `data.features[*].name` is the routing key for both dataset loading and model
  dispatch. Named features bind to `model.forward()` by keyword, so the entry
  name must match the forward parameter name.
- `data.features[*]` and `data.targets[*]` may omit `format` for
  loadable path-based entries when the path suffix is informative. The config
  layer infers `.npy`, `.npz`, `.csv`, `.txt`, `.parquet`, `.h5`, `.hdf5`,
  and `.zarr` before discriminated-union validation. Ambiguous paths should
  use an explicit `format = "..."`.
- Component `module_path` values remain optional; when provided they are
  validated at config load time via module discovery without executing the
  target module body, and runtime builders still apply default module
  namespaces when omitted.
- `ModelComponentSettings.name` uses `validation_alias="class"` (with
  `populate_by_name=True`) so TOML uses `class = "MyModel"` while `name=` still
  works as a Python kwarg; providing both `name` and `class` raises `ValueError`
  at validation time.
- Nested `[model.params]` is not supported. Model hyperparameters live directly
  under `[model]`, and HPO paths target `model.<field>` rather than
  `model.params.<field>`.
