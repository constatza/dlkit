# Runtime Workflow Factories

`dlkit.engine.workflows.factories` owns runtime component construction.

## Responsibilities
- choose the correct build strategy for the dataset/model family
- assemble datasets, datamodules, feature pipelines, and trainers
- create tracking-aware training executors for the training path

## Current Layout
- `build_factory.py`: dispatcher and public re-export surface
- `build_strategy.py`: shared strategy protocol and graph strategy
- `flexible_build_strategy.py`: flexible-array strategy
- `generative_build_strategies.py`: generative/flow-matching strategy
- `dataset_builder.py`: runtime dataset and datamodule assembly, plus split
  resolution split by responsibility: `build_split_for_training` (producer —
  training/optimize/converge) and `build_split_for_evaluation` (consumer —
  inference/evaluation; never generates a split)
- `run_output_paths.py`: `resolve_local_artifact_root()`, the single source
  of truth for "where does this run's local output live" — used by both
  split persistence and checkpoint dirpath pinning (`build_strategy.py`)
- `datamodule_resolution.py`: shared `DataModuleSelector` -> `LightningDataModule`
  resolution, used by both `dataset_builder.py` (training) and
  `inference_data_factory.py` (inference)
- `inference_data_factory.py`: inference-only datamodule assembly, reusing
  `DatasetBuilder` and `datamodule_resolution.py` rather than duplicating them
- `feature_pipeline.py`: feature/target transform assembly
- `execution_strategy_factory.py`: training executor composition with tracking activation
- `tracking_flag.py`: `apply_mlflow_flag()` — ensures MLflow tracking is configured
  on settings before dispatch. Lives here (not in `entrypoints`, its main caller)
  so `engine.workflows.multi_run.orchestrator` can depend on it too without a
  disallowed `engine.workflows -> entrypoints` edge (`tach.toml` only allows the
  reverse); re-exported from `entrypoints/__init__.py` for its other callers.

## Notes
- Dataset-family selection delegates to `runtime.data.families.resolve_family`.
- `DATASET.family` short-circuits family heuristics when explicitly configured.
- Flexible dataset assembly consumes explicit `DATASET.features` and `DATASET.targets` only.
- Flexible contract inference delegates feature and target shape propagation to `engine.data.geometry` from a single sampled item.
- Graph dataset assembly forwards `DATASET.root` into PyG dataset constructors so processed caches do not fall back to PyG's `???` placeholder root on Windows.
- Runtime builders, not `tools.config`, own default module-path resolution.
- Split resolution is split by responsibility (Interface Segregation), not by
  an optional flag on one shared method: `resolve_training_split` (producer)
  seeds via `settings.run.resolve_seed()` and, when a local
  `training.trainer.default_root_dir` is configured, always persists the
  resolved split under `<root>/splits/`; `resolve_evaluation_split`
  (consumer) has no ratio/seed parameters at all and can only reload a split
  file — it structurally cannot regenerate one. Evaluation resolves that
  file from an explicit `data.splits.filepath`, or auto-locates a colocated
  `splits/*.json` next to the checkpoint's run root (the checkpoint actually
  supplied — a caller-provided override always takes precedence over
  `settings.model.checkpoint`, matching `load_model_from_settings`'s
  existing override precedence). Both branches fail loudly
  (`ConfigurationError`) rather than silently falling back to generating an
  unrelated random split — this is the fix for a real bug where `evaluate()`
  previously regenerated a fresh, unseeded split on every call, uncorrelated
  with the model's actual held-out test rows. `data.splits.test_ratio`/
  `val_ratio` are also rejected as dead config in evaluation mode, since the
  split is always reloaded verbatim.
- Build strategies now attach typed split-artifact metadata to
  `RuntimeComponents.artifacts` so tracking can publish the exact split used by
  the run without reading datamodule ad hoc attributes.
- In MLflow-off mode, DLKit treats local-output-producing trainer features as
  opt-in. Checkpointing, local loggers, and `ModelCheckpoint` callbacks require
  an explicit `TRAINING.trainer.default_root_dir`, and all Lightning-owned
  local writes are pinned under it.
- In NoOp mode, when checkpointing is disabled and no local-output-producing
  trainer components are configured, `default_root_dir` is not required and
  trainer construction should not create local output directories.
- MLflow tracking is enabled only by explicit `tracking.backend = "mlflow"` configuration.
- When MLflow is enabled, durable artifacts belong to MLflow.
- `DataModuleSelector` is a `ComponentSettings`, resolved via
  `FactoryProvider.create_component()` like every other component —
  `dataset`/`split`/`dataloader` flow in through `BuildContext.overrides`.
  Both training and inference call the shared
  `datamodule_resolution.build_datamodule_from_selector()`, which also
  substitutes a dataset-family default class (e.g. `GraphDataModule`) into
  the selector before construction when the family implies one and the user
  hasn't picked an explicit class.
- Inference-only datamodule construction stays flexible/array-only: it always
  builds a `FlexibleDataset` via `DatasetBuilder`/`flexible_dataset_overrides`,
  with no graph-family dataset path (`graph_dataset_overrides` is
  training/`GraphBuildStrategy`-only).
