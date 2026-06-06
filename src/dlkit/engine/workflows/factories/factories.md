# Runtime Workflow Factories

`dlkit.engine.workflows.factories` owns runtime component construction.

## Responsibilities
- choose the correct build strategy for the dataset/model family
- assemble datasets, datamodules, feature pipelines, and trainers
- create tracking-aware training executors for the training path

## Current Layout
- `build_factory.py`: dispatcher and public re-export surface
- `build_strategy.py`: shared strategy protocol and graph/timeseries strategies
- `flexible_build_strategy.py`: flexible-array strategy
- `generative_build_strategies.py`: generative/flow-matching strategy
- `dataset_builder.py`: runtime dataset and datamodule assembly
- `feature_pipeline.py`: feature/target transform assembly
- `execution_strategy_factory.py`: training executor composition with tracking activation

## Notes
- Dataset-family selection delegates to `runtime.data.families.resolve_family`.
- `DATASET.family` short-circuits family heuristics when explicitly configured.
- Flexible dataset assembly consumes explicit `DATASET.features` and `DATASET.targets` only.
- Flexible contract inference delegates feature and target shape propagation to `engine.data.geometry` from a single sampled item.
- Graph dataset assembly forwards `DATASET.root` into PyG dataset constructors so processed caches do not fall back to PyG's `???` placeholder root on Windows.
- Runtime builders, not `tools.config`, own default module-path resolution.
- Split generation is seeded before runtime build and remains in memory unless an
  explicit `DATASET.split.filepath` is provided.
- Build strategies now attach typed split-artifact metadata to
  `RuntimeComponents.artifacts` so tracking can publish the exact split used by
  the run without reading datamodule ad hoc attributes.
- When MLflow is disabled, trainer construction pins Lightning-owned local
  writes under `TRAINING.trainer.default_root_dir`; when MLflow is enabled,
  durable artifacts belong to MLflow.
