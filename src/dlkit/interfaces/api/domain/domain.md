# API Domain Module

`dlkit.interfaces.api.domain` contains API-local boundary types only.

## Lives Here
- `TrainingOverrides`
- `OptimizationOverrides`
- `ExecutionOverrides`
- `RuntimeOverrideModel`
- small interface/test protocols
- precision helper re-exports that are still relevant at the API boundary

## Notes
- Override payloads are strict Pydantic models with `extra="forbid"` so typoed keys fail immediately instead of being ignored.

## Does Not Live Here
Workflow results, model state, and shared errors come from `dlkit.common`.
