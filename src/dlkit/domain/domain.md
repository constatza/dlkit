# Domain Layer

`dlkit.domain` owns pure model-adjacent logic with no runtime orchestration.

## Current Modules
- `nn`
- `shapes`
- `transforms`
- `metrics`
- `losses`

## Shapes
The shape subsystem is split into focused modules:
- `inference_engine.py`
- `inference_strategies.py`
- `specification_base.py`
- `shape_specifications.py`
- `validation_engine.py`
- `serialization_types.py`
- `shape_serializer.py`
- `shape_migrator.py`

Dataset-driven shape inference normalizes scalar tensors to `(1,)` so class-label
targets and other zero-rank samples produce valid shape entries instead of
falling back to defaults.
Serialized shape payloads use the versioned metadata wrapper; runtime deserialization
expects that canonical format.

## Transforms
The transform pipeline surface uses `TransformContext` as the runtime-facing
value object around feature/target transform chains.

These modules provide value-level and algorithmic behavior that runtime services
can consume without pulling orchestration back into the domain layer.
