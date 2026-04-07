# Runtime Workflow Factories

`dlkit.engine.workflows.factories` owns runtime component construction.

## Responsibilities
- choose the correct build strategy for the dataset/model family
- assemble datasets, datamodules, feature pipelines, and trainers
- run runtime shape inference when required
- create tracking-aware training executors for the training path

## Current Layout
- `build_factory.py`: dispatcher and public re-export surface
- `build_strategy.py`: shared strategy protocol and graph/timeseries strategies
- `flexible_build_strategy.py`: flexible-array strategy
- `generative_build_strategies.py`: generative/flow-matching strategy
- `dataset_builder.py`: runtime dataset and datamodule assembly
- `feature_pipeline.py`: feature/target transform assembly
- `shape_inference_pipeline.py`: runtime shape inference coordination
- `execution_strategy_factory.py`: training executor composition with tracking activation

## Notes
- Dataset-family selection delegates to `runtime.data.families.resolve_family`.
- `DATASET.family` short-circuits family heuristics when explicitly configured.
- Flexible dataset assembly consumes explicit `DATASET.features` and `DATASET.targets` only.
- Runtime builders, not `tools.config`, own default module-path resolution.
