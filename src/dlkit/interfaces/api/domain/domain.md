# API Domain Module

`dlkit.interfaces.api.domain` contains API-local boundary types only.

## Lives Here
- `TrainingOverrides`
- `OptimizationOverrides`
- `ExecutionOverrides`
- small interface/test protocols
- precision helper re-exports that are still relevant at the API boundary

## Does Not Live Here
Workflow results, model state, and shared errors come from `dlkit.shared`.
