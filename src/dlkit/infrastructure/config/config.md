# Configuration Module

`dlkit.infrastructure.config` owns typed settings, validation, patching,
workflow-specific config views, and component-setting models.

## Responsibilities

- immutable Pydantic settings models (`frozen=True`)
- `JobConfig` top-level discriminated union (training / inference / search / convergence / multirun / fit); `AnyJobConfig` is the PEP 695 sum-type alias of every validated subtype
- TOML loading via `load_job()` with deep-merge and profile references
- patch application and runtime override support
- component settings and factory support
- security-oriented URI and path config types
- load-time validation for importable component `module_path` values

Precision is documented in [`../precision/precision.md`](../precision/precision.md).
Seeding is documented in [`../seeding/seeding.md`](../seeding/seeding.md).

## Current Structure

- `core/`: base settings, patching, factories, build context, TOML source
- `core/_path_helpers.py`: path-preprocessing helpers (training / model / data)
- `job_config.py`: `JobConfig`, `TrainingJobConfig`, `InferenceJobConfig`, `SearchJobConfig`, `FitJobConfig`, `ConvergenceJobConfig`, `MultiRunJobConfig`, `ChildEntryConfig`, `AnyJobConfig` (sum type)
- `run_settings.py`: `RunSettings` (type, seed, precision, profile references);
  `RunSettings.resolve_seed()` (pure seed defaulting) and the module-level
  `apply_run_context(run)` context manager (seeds global RNG state and
  applies the run's precision override — see
  [`../seeding/seeding.md`](../seeding/seeding.md))
- `plot_settings.py`: `PlotSettings` (opt-in plot artifact generation)
- `experiment_settings.py`: `ExperimentSettings` (name, run_name, tags)
- `model_components.py`: canonical `ModelComponentSettings`, plus loss/metric component settings
- `data_settings.py`: `DataSettings` plus entry types
- `training_settings.py`: `TrainingSettings`, `StoppingSettings`
- `search_settings.py`: `SearchSettings`, param types
- `convergence_settings.py`: sample-size convergence settings
- `tracking_settings.py`: `TrackingSettings`
- `optimizer_policy.py`: `OptimizerPolicySettings`
- `optimizer_component.py`: concrete optimizer and scheduler component settings

## Tracking Settings

`tracking.model_serialization_format` controls how PyTorch model artifacts are
logged to MLflow:

- `"pickle"` is the default compatibility format.
- `"pt2"` opts into MLflow's `torch.export`-backed PyTorch serialization for
  deployment-oriented artifacts. PT2 export requires input-shape metadata so the
  tracking layer can provide an `input_example`.

Lightning `.ckpt` files remain checkpoint artifacts under `checkpoints/`; they
are separate from the logged deployment model under `model/`.

`tracking.max_retries` sets MLflow's process-wide HTTP retry budget
(`environment.ensure_mlflow_defaults`/`set_mlflow_max_retries`, default 5
retries / 30s timeout / backoff factor 2) for calls that must not silently
fail, e.g. the logged deployment model artifact. Tracking calls that are
allowed to fail (metrics, params, tags, non-critical artifacts — anything
wrapped in `engine.tracking.best_effort`) run under a separate, tighter
fail-fast budget scoped to just that call
(`environment.best_effort_retry_budget`, 2 retries / 5s timeout / backoff
factor 1) so a persistently-erroring tracking server can't turn every trial
into a multi-second-to-multi-minute stall; `tracking.max_retries` is
untouched by that scoping.

## Loading a Config

```python
from dlkit.infrastructure.config.factories import load_job

job = load_job("config.toml")                # type inferred from run.type
job = load_job(["base.toml", "local.toml"]) # merged left-to-right
job = load_job("config.toml", run_type="train")  # override type
```

`[run]` may reference a separate profile TOML file for `model`, `data`,
`training`, `tracking`, or `plots` (e.g. `run.plots = "../profiles/plots.toml"`).
Each referenced file must contain a top-level section matching the key name;
its content is merged as the base for that section, with the job file's own
section taking precedence.

## Data Splits

`data.splits` (`IndexSplitSettings` in `split_settings.py`) controls how a
dataset is partitioned into train/validation/test/predict index sets. Two
mutually exclusive modes:

- **Ratio mode** (default): give `test` and `val` fractions and a random split
  is generated at runtime.

  ```toml
  [data.splits]
  val = 0.15
  test = 0.15
  ```

- **External file mode**: give `filepath` pointing at a pre-computed split
  file (JSON or TOML, dispatched by extension). When set, `filepath` takes
  precedence over `test`/`val` entirely — no random split is generated.

  ```toml
  [data.splits]
  filepath = "splits/my_split.toml"
  ```

  The file holds flat index lists; `predict` is optional:

  ```toml
  train = [0, 1, 2, 3]
  validation = [4, 5]
  test = [6, 7]
  ```

  Relative `filepath` values resolve against the config file's own directory,
  same as other dataset-owned paths (see Path Ownership below).

`max_train_samples` and `train_subset_seed` optionally cap/re-permute the
resolved train split afterward (both modes), for convergence studies that
need nested training subsets — see `engine/data/data.md`.

## Multirun Settings

`run.type = "multirun"` selects `MultiRunJobConfig`. Authored in TOML under a
single `[multirun]` table:

```toml
[run]
type = "multirun"

[multirun]
experiment_name = "sweep"
parent_run_name = "sweep-parent"
parent_tags = { team = "platform" }              # optional, default {}
failure_policy = "continue_mark_parent_failed"   # optional, default "fail_fast"

[[multirun.runs]]
id = "a"
label = "Run A"
files = ["jobs/base.toml", "jobs/variant_a.toml"]   # merged left-to-right
patches = { "run.seed" = 7 }                        # optional, applied after loading

[[multirun.runs]]
id = "variants"
files = "jobs/variants/*.toml"   # a string containing *, ?, or [ is glob shorthand
```

`MultiRunJobConfig` hoists the `[multirun]` table's keys onto itself (a
`model_validator(mode="before")`) so callers read `settings.experiment_name`,
`settings.parent_run_name`, `settings.parent_tags`, `settings.failure_policy`,
and `settings.runs` directly — no `.multirun` indirection. Each
`[[multirun.runs]]` entry is a `ChildEntryConfig` (`id`, `label`, `files`,
optional `run_type`, optional `patches`, opaque `tags`/`params`/`metadata`).
`patches` is rejected on a glob-sourced entry (`files` containing `*`/`?`/`[`)
— a single patch dict can't sensibly apply to every glob match.

An optional `[multirun.defaults]` table (`ChildDefaults`: `patches`, `tags`,
`params`, `metadata`) is deep-merged under every `[[multirun.runs]]` entry's
own matching field, child values winning on conflict — opt-in only, empty by
default, so a sweep with no `[multirun.defaults]` table shares nothing
implicitly across children:

```toml
[multirun.defaults]
tags = { team = "platform" }
patches = { "run.precision" = "float32" }

[[multirun.runs]]
id = "a"
files = "jobs/a.toml"
tags = { dataset = "ds-1" }   # merges with defaults.tags -> {team, dataset}
```

The merge (`engine.workflows.entrypoints.multirun._apply_defaults()`, using
the same `deep_merge()` `load_job()` uses for TOML-file/profile/env-patch
merging) happens before `ChildEntryConfig -> ChildSource` conversion, so
`build_child_sources()` itself stays defaults-unaware.

`load_job()` resolves every child entry's `files` (explicit list or glob
pattern) to an absolute path/pattern relative to the multirun config file's
own directory, the same convention profile references use — by the time a
`ChildEntryConfig` is validated, `files` holds only absolute values. The
`ChildEntryConfig -> ChildSource` conversion (glob-vs-explicit dispatch,
`GlobSource`/`ExplicitFileSource` construction) lives in the engine layer
(`engine.workflows.entrypoints.multirun.build_child_sources()`), not here —
infrastructure must not import engine per the DAG.

## Convergence Settings

`run.type = "converge"` selects `ConvergenceJobConfig`. The `convergence`
section accepts either explicit `sizes` or `min_samples` / `max_samples` /
`steps` for log-spaced sample-size generation. `repeats` controls independent
runs per size, and optional `target`, `target_metric`, and `c` fields control
threshold-based `n_star` detection.

## Fit Settings

`run.type = "fit"` selects `FitJobConfig` — `model` and `data` are required,
same as `TrainingJobConfig`, but `training` is intentionally left unset
rather than required. For models whose entire "training" is one
deterministic, non-gradient call (e.g. a thin-SVD basis fit into a
`register_buffer`) — no epochs, optimizer, or loss, so nothing downstream
should ever need to route the model through optimizer/loss wiring. Dispatched
by `engine.workflows.factories.fit_build_strategy.FitBuildStrategy` (build)
and `engine.training.one_shot_fit_executor.OneShotFitExecutor` (execute); see
`engine.training.execution.execution.md`.

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
