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

## Transforms
The transform pipeline surface uses `TransformContext` as the runtime-facing
value object around feature/target transform chains.

These modules provide value-level and algorithmic behavior that runtime services
can consume without pulling orchestration back into the domain layer.
