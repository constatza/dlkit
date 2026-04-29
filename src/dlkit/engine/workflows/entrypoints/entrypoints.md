# Runtime Entrypoints

`dlkit.engine.workflows.entrypoints` contains the runtime-owned workflow edge.

## Responsibilities
- coerce workflow settings to runtime-ready settings objects
- accept strict Pydantic override payloads
- validate and apply request-scoped overrides
- establish path override context
- measure elapsed time for workflow results
- own optimization tracker context lifecycle at the runtime edge

## Current Layout
- `_settings.py`: workflow settings coercion
- `_override_types.py`: strict override payload models
- `_entrypoint_context.py`: shared setup for override application, path context, and timing
- `training.py`: training entrypoint
- `optimization.py`: optimization entrypoint
- `execution.py`: training-vs-optimization routing
- `validation.py`, `templates.py`, `convert.py`: validation/template/export helpers

## Design Rule
Entrypoints stay procedural. They normalize request-level concerns and then hand
control to runtime orchestration and optimization services.

Unknown override keys are rejected at the entrypoint boundary instead of being silently dropped.
