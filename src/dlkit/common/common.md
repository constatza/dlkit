# Common Module

`dlkit.common` contains the lowest-level contracts reused across the rest of
the codebase.

## Responsibilities
- shared workflow result types
- shared error hierarchy
- shared model state contracts
- geometry primitives (FieldRole, GeometryKind, FieldSpec, GeometrySpec)
- cross-layer lifecycle hooks

## Current Contracts
- Errors: `DLKitError`, `ConfigurationError`, `WorkflowError`, `StrategyError`, `ModelStateError`, `PluginError`, `ModelLoadingError`
- Results: `TrainingResult`, `InferenceResult`, `OptimizationResult`
- Geometry: `FieldRole`, `GeometryKind`, `TopologyKind`, `FieldSpec`, `GeometrySpec`
- Hooks: `LifecycleHooks`

`TrainingResult` includes lazy derived accessors for prediction payloads through:
- `stacked`
- `to_numpy()`

## Design Rule
`shared` does not import higher layers. Runtime orchestration, config loading,
and model construction stay out of this package.
