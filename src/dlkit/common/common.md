# Common Module

`dlkit.common` contains the lowest-level contracts reused across the rest of
the codebase.

## Responsibilities
- shared workflow result types
- shared error hierarchy
- shared model state contracts
- geometry primitives (FieldRole, GeometryKind, FieldSpec, GeometrySpec)
- cross-layer lifecycle hooks
- thin structural protocols for settings contracts (no infrastructure imports)

## Current Contracts
- Errors: `DLKitError`, `ConfigurationError`, `WorkflowError`, `StrategyError`, `ModelStateError`, `PluginError`, `ModelLoadingError`
- Results: `TrainingResult`, `InferenceResult`, `OptimizationResult`
- Geometry: `FieldRole`, `GeometryKind`, `TopologyKind`, `FieldSpec`, `GeometrySpec`
- Hooks: `LifecycleHooks`
- Hook param scalars: `ParamValue = str | int | float | bool`
- Protocols: `IDataModule`, `ITrainableModule`

`TrainingResult` includes lazy derived accessors for prediction payloads through:
- `stacked`
- `to_numpy()`

## Geometry: roles and spatial structure

`FieldRole`, `GeometryKind`, `FieldSpec`, and `GeometrySpec` form the geometry
vocabulary used by the engine's shape-inference and contract-resolution pipelines.

### FieldRole — physics semantics

| Role | Meaning | Drives |
|------|---------|--------|
| `FEATURE` | Primary model input (sensor data, node features, etc.) | `in_shape` / `in_channels` in contracts |
| `FEATURE_COORDINATES` | Spatial coordinates that accompany the feature (e.g. query grid) | `in_channels` or `spatial_shape` depending on geometry kind |
| `TARGET_COORDINATES` | Query coordinates for the output evaluation | `query_shape` in `BranchTrunkSpec` |

### GeometryKind — spatial structure

| Kind | Tensor layout | Contract produced |
|------|--------------|-------------------|
| `TABULAR` | `(features,)` | `TabulaRSpec` |
| `SEQUENCE` | `(channels, seq_len)` | `SequenceSpec` |
| `REGULAR_GRID` | `(channels, *spatial_dims)` | `GridOperatorSpec` |
| `POINT_CLOUD` | `(channels, coord_dim)` | `GridOperatorSpec` |
| `GRAPH` | `(node_features,)` | `GraphContractSpec` |

`TABULAR` with `TARGET_COORDINATES` fields → `BranchTrunkSpec`.

### Building a GeometrySpec

```python
from dlkit.common.geometry import FieldRole, GeometryKind, FieldSpec, GeometrySpec

geometry = GeometrySpec(
    fields=(
        FieldSpec(name="x", shape=(16,), role=FieldRole.FEATURE),
        # add TARGET_COORDINATES for DeepONet / branch-query networks:
        # FieldSpec(name="query", shape=(3,), role=FieldRole.TARGET_COORDINATES),
    )
)
```

`GeometrySpec` is normally inferred automatically by the engine from `entry_configs`
and dataset samples. Construct it manually only for advanced use cases or tests.

`GeometrySpec.get_shape(name)` looks up a field by name and returns its shape tuple,
or `None` if not found. This is the interface consumed by `ShapeAwareTransform.configure_shape`
for eager buffer pre-allocation.

## Design Rule
`common` does not import higher layers. Runtime orchestration, config loading,
and model construction stay out of this package.
