# Runtime Entrypoints

`dlkit.engine.workflows.entrypoints` contains the runtime-owned workflow edge.

## Responsibilities
- coerce workflow settings to runtime-ready settings objects
- accept strict Pydantic override payloads
- validate and apply request-scoped overrides
- establish path override context
- measure elapsed time for workflow results
- own experiment-tracker lifecycle at the runtime edge
- delegate optimization backend-session lifecycle to runtime orchestrators

## Current Layout
- `_settings.py`: workflow settings coercion
- `_override_types.py`: strict override payload models
- `_entrypoint_context.py`: shared setup for override application, path context, and timing
- `training.py`: training entrypoint
- `optimization.py`: optimization entrypoint
- `convergence.py`: sample-size convergence study entrypoint
- `multirun.py`: general multirun sweep entrypoint (`run_multirun()` from a
  `MultiRunJobConfig`, `run_multirun_spec()` from an already-built
  `MultiRunSpec`); also owns the `ChildEntryConfig -> ChildSource` conversion
  (`build_child_sources()`, glob-vs-explicit decided by whether `files`
  contains a glob metacharacter)
- `execution.py`: training-vs-optimization routing
- `validation.py`, `templates.py`, `convert.py`: validation/template/export helpers

## Design Rule
Entrypoints stay procedural. They normalize request-level concerns and then hand
control to runtime orchestration and optimization services. They may enter
top-level tracker contexts when a workflow needs runtime-owned tracking setup,
but they do not enter optimization backend-session contexts themselves.

Convergence entrypoints validate convergence-specific overrides, build
sample-size training sweeps, and delegate repeat execution to engine
orchestrators before returning aggregated convergence points.

Unknown override keys are rejected at the entrypoint boundary instead of being silently dropped.

`apply_mlflow_flag()` (ensures MLflow tracking is configured before dispatch)
lives in `engine.workflows.factories.tracking_flag`, not here, and is
re-exported from this package's `__init__.py`. It has two other consumers —
`interfaces.api.adapters.workflow_executor` and
`engine.workflows.multi_run.orchestrator` — and `tach.toml` only allows
`entrypoints -> engine.workflows` (not the reverse), so it can't live in a
module the multirun orchestrator itself is disallowed from importing.

`multirun.py` needs the identical deferred-import trick `convergence.py`
uses (`from .execution import execute as dispatch_execute` inside the
function body, not at module level) for the same reason: `execution.py`
imports `converge` from this package at module level, so a module-level
`from .execution import execute` here would risk the same cycle.

This package also re-exports `MultiRunSpec`, `RunSpec`, and
`expand_child_sources` from `engine.workflows.multi_run` — not because they
belong here, but because `dlkit.interfaces.cli`/`dlkit.interfaces.api.*` are
only allowed to depend on `engine.workflows.entrypoints`/`.factories` per
`tach.toml`, not the general `engine.workflows.multi_run` bucket, so those
upper layers get them through this package's re-export instead of importing
`multi_run` directly.

The parent tracker for a multirun sweep (`multirun.py`'s `_run_sweep()`) is
configured from the **first child's own settings**, mlflow-flag-forced —
mirroring `MultiRunOrchestrator._run_one()`'s existing per-child behavior.
A multirun sweep's entire purpose is parent/child MLflow linkage, so tracking
is always enabled rather than silently producing an untrackable sweep; the
public `run_multirun_config()`/`run_multirun_spec()` API functions accept an
`mlflow: bool` parameter for signature symmetry with `train()`/`converge()`/
`optimize()`, but it has no effect for multirun.

`run_multirun()`/`run_multirun_spec()` accept an optional `hooks:
LifecycleHooks | None` and forward it into `MultiRunOrchestrator`, which
forwards it into every child's own `execute()` call — a caller's
`on_run_created` fires for the parent sweep run and for every child's own
run. `ChildEntryConfig.tags` (TOML) / `RunSpec.tags` (programmatic) reach the
child's actual MLflow run via `ExperimentSettings.tags`, merged with
whatever the child's own settings already set (child tags win on
conflict) — this is what makes tag-filtered run lookup work for a sweep's
children. `MultiRunJobConfig.parent_tags` / `MultiRunSpec.parent_tags` reach
the parent run the same way. A `ChildFailure`'s `run_id` is a best-effort
capture of whatever run the child opened before it raised (via a wrapped
`on_run_created`), not guaranteed — a child that fails before opening any
run still reports `run_id=None`.

## Two-phase dispatch pattern (no dispatch injection seam)

`_run_sweep()` always dispatches children through the runtime's own
`execute()` — there is no caller-supplied dispatcher parameter. A caller
needing work *around* each child's execution (e.g. staging a dataset before
training, or moving a checkpoint to durable storage after) does it in two
phases instead of injecting into dispatch:

1. **Before calling `run_multirun_config()`/`run_multirun_spec()`**: do any
   per-child prep (dataset staging, etc.) while building each child's
   settings/`RunSpec`.
2. **After the call returns**: iterate the returned `MultiRunResult`'s
   `ChildOutcome`s, keyed by `child_id`, and do post-processing (checkpoint
   durability, registry updates) against each child's `run_id`/`result`
   (present on `ChildSuccess`; `run_id` is also present, best-effort, on
   `ChildFailure`).

This needs no DLKit-side seam: everything a caller needs per child (`tags`,
`run_id`, the raw workflow `result`) is already on the returned
`ChildOutcome`.
