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
- `_entrypoint_context.py`: shared setup for override application, path context, and timing.
  `EntrypointContext.run(workflow_fn, error_message=..., error_class=WorkflowError)` is the
  single place all five entrypoints (`fit.py`, `training.py`, `optimization.py`,
  `convergence.py`, `evaluate.py`) route their workflow execution through: it runs
  `workflow_fn` under `run_with_path_context`, re-raises a `WorkflowError` unchanged, and
  wraps any other exception via `raise_error(error_message, exc, error_class=error_class)`.
- `training.py`: training entrypoint
- `fit.py`: one-shot, non-gradient fit entrypoint (`FitJobConfig`). Unlike
  trainer-backed jobs, `FitJobConfig` has no `training.trainer.default_root_dir`
  knob, so `FitOverrides.checkpoints_dir` gives callers running many fit jobs
  from the same cwd (e.g. a batch of assignments) a way to isolate each
  call's checkpoint write path: when set, `fit()` wraps the orchestrated
  execution in `path_override_context({"checkpoints_dir": ...})` so
  `OneShotFitExecutor`'s fixed `checkpoints/fitted.ckpt` path resolves under
  a distinct directory per call instead of colliding.
- `optimization.py`: optimization entrypoint
- `convergence.py`: sample-size convergence study entrypoint
- `evaluate.py`: evaluation entrypoint — checkpoint + labeled dataset ->
  metrics + figures, no training loop. A fourth peer of
  training/optimization/convergence, not a special case: it fits the same
  `ChildDispatcher` shape (`(settings, overrides=None, *, hooks=None) ->
  result`) and is dispatched by `execute()`/`MultiRunOrchestrator` exactly
  like the other three. Lives here (not `interfaces.inference`, where it
  used to live) because it needs `engine.tracking` and
  `engine.workflows.factories`, which only this package — not
  `engine.inference` — is allowed to reach.
- `multirun.py`: general multirun sweep entrypoint (`run_multirun()` from a
  `MultiRunJobConfig`, `run_multirun_spec()` from an already-built
  `MultiRunSpec`); also owns the `ChildEntryConfig -> ChildSource` conversion
  (`build_child_sources()`, glob-vs-explicit decided by whether `files`
  contains a glob metacharacter). `run_multirun()` first runs every entry
  through `_apply_defaults()`, deep-merging `MultiRunJobConfig.defaults`
  (the optional `[multirun.defaults]` TOML table) under each entry's own
  `patches`/`tags`/`params`/`metadata`, child values winning — opt-in only,
  empty by default, so nothing is shared across children unless a sweep
  author explicitly writes that table.
- `execution.py`: routes settings to whichever of train/optimize/converge/evaluate applies
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

This package also re-exports `MultiRunSpec`, `RunSpec`, `ExistingRunsSource`,
and `expand_child_sources` from `engine.workflows.multi_run` — not because
they belong here, but because `dlkit.interfaces.cli`/`dlkit.interfaces.api.*`
are only allowed to depend on `engine.workflows.entrypoints`/`.factories` per
`tach.toml`, not the general `engine.workflows.multi_run` bucket, so those
upper layers get them through this package's re-export instead of importing
`multi_run` directly. `ExistingRunsSource` is a `ChildSource` variant that
expands one evaluate `RunSpec` per active child of an existing parent run
(via `find_child_run_ids()`), each pointed at that child's own checkpoint —
this is what `interfaces.api.functions.core.evaluate_multirun()` composes to
replace the old bespoke `evaluate_multirun()` fan-out. `ExistingRunsSource.settings`
(and `evaluate_multirun()`'s `settings` param) accepts either a single
`InferenceJobConfig` shared verbatim across every child, or a
`Callable[[str], InferenceJobConfig]` keyed by each child's own run id for
sweeps that need to vary more than the checkpoint (e.g. different
datasets/models per child) — nothing beyond the checkpoint is shared unless
a caller passes the plain-object form explicitly. The callable form
requires `tracking_uri` to be set explicitly (there is no single settings
object to default it from before child run ids are known).

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
run still reports `run_id=None`. `on_child_planned` fires right before each
child dispatches (settings finalized, no MLflow run created yet);
`on_child_completed` fires with the child's `ChildSuccess` right after it
succeeds, paired with `on_child_failed` on the failure path. Both hooks, and
the `label`/`params`/`metadata` fields on `ChildSuccess`/`ChildFailure`,
require zero changes anywhere in this file's own code — `LifecycleHooks`
already flows through opaquely end to end, so new fields on it are
automatically reachable from the public API with no plumbing changes.

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
