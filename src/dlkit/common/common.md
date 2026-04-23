# Common Module

`dlkit.common` contains the lowest-level contracts reused across the rest of
the codebase.

## Responsibilities
- shared workflow result types
- shared error hierarchy
- shared model state contracts
- minimal shape protocols and summaries
- cross-layer lifecycle hooks

## Current Contracts
- Errors: `DLKitError`, `ConfigurationError`, `WorkflowError`, `StrategyError`, `ModelStateError`, `PluginError`, `ModelLoadingError`
- Results: `TrainingResult`, `InferenceResult`, `OptimizationResult`
- Shapes: `ShapeSpecProtocol`, `ShapeSummary` (validates non-empty in_shapes and out_shapes at construction)
- Hooks: `LifecycleHooks`

`TrainingResult` includes lazy derived accessors for prediction payloads through:
- `stacked`
- `to_numpy()`

## Design Rule
`shared` does not import higher layers. Runtime orchestration, config loading,
and model construction stay out of this package.
