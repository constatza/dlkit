# Optimization Module

`dlkit.engine.workflows.optimization` is the runtime-owned subsystem for the
hyperparameter-optimization workflow. The package root is a marker only; import
concrete modules directly.

## Overview

The optimization workflow is a runtime-owned hexagonal subsystem.
Its public runtime entrypoint is `dlkit.engine.workflows.entrypoints.optimize()`.

## Internal Layers

### Domain

Pure optimization concepts and contracts:
- study and trial models
- optimization result model
- study repository, backend-session, and tracking protocols

### Application

Orchestration services that coordinate:
- study lifecycle
- optimization backend-session lifecycle
- trial execution
- configuration preparation
- interaction with runtime build components

### Infrastructure

Adapters for external systems:
- Optuna persistence
- Optuna backend sessions
- shared backend-study registry for Optuna repository/session internals
- MLflow tracking
- configuration serialization

`optimize()` accepts the same `dlkit.common.hooks.LifecycleHooks` used by `train()`,
and all five hooks fire, not just `on_run_created`. `MLflowTrackingAdapter` fires
`on_run_created` with a `dlkit.common.hooks.RunCreatedEvent(run_id, tracking_uri,
kind, is_outermost)` for the study run (`kind="study"`), each trial run
(`kind="trial"`), and the best-retrain run (`kind="best_retrain"`). Only
`is_outermost=True` events (the study run) should be tagged to an externally-owned
parent run — `kind="trial"`/`"best_retrain"` runs are already correctly nested via
MLflow's own active-run stack and must not be re-tagged. Callers must read
`RunCreatedEvent.kind`/`.is_outermost` explicitly rather than relying on firing
order. `on_training_complete`/`extra_params`/`extra_tags`/`extra_artifacts` fire for
every trial (via `fire_post_training_hooks`, cheap callables only) and for the best
retrain (via `TrackingDecorator.execute_within_run`, which fires them as part of full
parity — see below).

Ordinary trials execute through `dlkit.engine.tracking.lightweight_execution.
execute_lightweight`: checkpoints disabled, epoch metrics only. This is an
intentional, named tradeoff for high-volume trial loops (settings/model TOML,
hyperparameters, and metrics are still logged per trial via module-level functions
in `infrastructure/tracking.py`) — not a lesser implementation of tracking, just a
lighter one. The best-retrain leg is different: because it's functionally a `train()`
call with fixed hyperparameters, `TrialExecutor.execute_best_retrain` reuses
`TrackingDecorator.execute_within_run` against the already-open retrain run context,
giving it the exact same plots/checkpoints/model-artifact/dataset-lineage logging a
plain `train()` call gets. `execute_within_run` never opens its own MLflow run (nested
run detection in `MLflowResourceManager` is per-tracker-instance, not global) — the
caller must already hold an open run on the same tracker instance passed to
`TrackingDecorator`; that's why `IStudyTracker.execution_tracker()` exists, to
hand the underlying tracker across this boundary.

There is no bespoke `ITrialRunContext`/`IStudyRunContext` hierarchy — run contexts are
the same `dlkit.engine.tracking.interfaces.IRunContext` (or `NullRunContext` when
tracking is disabled) used everywhere else in the tracking stack. Trial/study-specific
concerns with no generic equivalent (hyperparameter logging, objective/duration,
sampler/pruner metadata, best-trial TOML summary) are plain functions in
`infrastructure/tracking.py`, not methods on a wrapper class.

`log_trial_hyperparameters` logs sampled search-space keys to MLflow with only
their top-level section prefix stripped (`model.num_layers` -> `num_layers`,
`training.optimizer.lr` -> `optimizer.lr`); the prefix itself only exists to
route the value to the right `JobConfig` field at patch time and is dropped
here as UI noise, while the rest of the path is kept so sibling leaves under
different sections don't collide. `log_trial_settings` tags every trial run
with `mlflow_model_class` (same tag/convention as the main-run artifact
logger), since trial runs never go through that artifact-logging path and
would otherwise be unidentifiable by model class once the run name is
uninformative.

On the best-retrain run only, `SettingsLogger.log_model_parameters` (run
inside `execute_best_retrain`'s full `TrackingDecorator` path) already logs
the model's resolved `hparams` under the same stripped names
`log_trial_hyperparameters` would use for `model.`-prefixed search-space
keys, so `_retrain_best_trial` filters those out of the hyperparameters dict
it passes to `log_trial_hyperparameters` — otherwise the same param gets
logged twice on that one run. Regular trial runs (`execute_lightweight`,
which skips `SettingsLogger` entirely) are unaffected and keep logging every
key, including `model.`-prefixed ones.

The outer/study run also carries the best trial's full training result
(`log_best_trial_result`, called alongside `log_study_summary`/
`log_best_trial_settings`): metrics via `MetricLogger.log_all_metrics`
(unfiltered — no `MLflowEpochLogger` runs against the study run, unlike the
best-retrain's own nested run, so nothing else would log them there) and
artifacts via `log_trial_artifacts`. So the study run alone looks like a
regular training result — metrics, artifacts, checkpoint — without needing
to open the nested best-retrain child run.

The public `dlkit.common.results.OptimizationResult` returned by `optimize()`
wraps a `TrainingResult` in `.training_result` and duck-types as one via
`__getattr__` — `.metrics`, `.artifacts`, `.checkpoint_path`, etc. all
delegate there, so callers that expect a plain training result can use a
search's result the same way, while `.best_trial`/`.study_summary` (and its
own `.duration_seconds`, the *total* search duration) stay as first-class
fields. `.training_result` is `None` when every trial failed or was pruned
(no retrain ran); accessing a delegated attribute in that case raises
`AttributeError` rather than silently returning `None`.

`.mlflow_run_id`/`.mlflow_tracking_uri` identify the study's own MLflow run
(same pair `TrainingResult`/`ConvergenceResult`/`EvaluationResult` carry, same
reason: a caller needs both to independently locate the run afterward via
`MlflowClient(tracking_uri=...)`, not just the run id alone). Populated in
`OptimizationOrchestrationService._execute_with_tracking` from the study
`IRunContext`, guarded on `is_active()` — the study run context is a real
object even when MLflow tracking isn't configured (Null Object Pattern), so
both fields are `None` rather than the null context's `run_id=""` sentinel
leaking through.

## Known Limitations

- Multiple independent searches (several `SearchJobConfig`s) as children of one
  general multirun sweep are supported via `MultiRunOrchestrator`
  (`engine.workflows.multi_run`), which dispatches each child through
  `engine.workflows.entrypoints.execute()` — the same type-based routing
  `optimize()`/`train()`/`converge()` already use, not a training-only path.
  `OptimizationResult` carries `mlflow_run_id`/`mlflow_tracking_uri` (populated
  from the study run's own `IRunContext` in `_execute_with_tracking`), so a
  search child's study run is tagged `mlflow.parentRunId` under the sweep's
  parent exactly like a train or convergence child — see
  `engine.workflows.entrypoints.entrypoints.md`. Callers that need to sweep
  several `SearchJobConfig`s outside of a multirun config can still loop and
  call `optimize()` directly; sequential `optimize()` calls in the same process
  are safe to do this way regardless: each call's `MLflowResourceManager` fully
  drains MLflow's active-run state (Study, every Trial, and any Best-retrain
  run) on exit regardless of nesting depth, so a later call never inherits a
  leftover active run from an earlier one.

- **Concurrency model: trials execute strictly sequentially, in-process.**
  `OptimizationOrchestrator` drives one plain `for trial_number in
  range(study.target_trials):` loop via Optuna's `ask()`/`tell()` API — there
  is no `n_jobs`, no thread pool, no `SearchSettings` concurrency field.
  "Running multiple searches" today means launching several separate OS
  processes, each with its own independent `MLflowResourceManager`; that
  works, but every process's HTTP traffic lands on the same tracking backend,
  so a shared local `mlflow server` sees load proportional to however many
  processes are running at once. If that produces transient artifact-upload
  failures (a burst of 5xx from the local server under load exhausting
  MLflow's HTTP retry budget), raise `TrackingSettings.max_retries` — it's
  wired through to `MLFLOW_HTTP_REQUEST_MAX_RETRIES` for the run.

  A future contributor adding real in-process concurrency (a `ThreadPoolExecutor`
  around the trial loop, using Optuna's thread-safe `ask()`/`tell()`) would need
  to address two shared-mutable-state blockers first: the global
  `mlflow.set_tracking_uri()` toggle on every `MLflowResourceManager.create_run()`
  entry/exit (races across threads sharing one manager), and the unsynchronized
  `_trial_mapping`/`_reported_trials`/`_active_storages` instance dicts on
  `OptunaOptimizationBackendSession`.

Optimization configuration persistence is opt-in for local files. When an
active tracker is available, small config artifacts should be logged through the
tracking boundary instead of creating implicit durable files on disk.

## Runtime Boundary

Runtime callers should use:
- `dlkit.engine.workflows.entrypoints.optimize()`
- `dlkit.engine.workflows.strategy.OptimizationStrategy`
- concrete imports from `domain`, `application`, or `infrastructure` when needed

The optimization orchestrator is responsible for:
- entering and exiting `IOptimizationBackendSession`
- entering and exiting the top-level experiment tracker when tracking is
  enabled, mirroring `MultiRunOrchestrator.run_sweep`'s `with self._tracker:`
- coordinating backend-specific sampling and reporting through that session
- entering tracker-owned nested run contexts after the backend session and
  tracker are both active

The runtime entrypoint is responsible for:
- applying request-level overrides
- managing path context
- calling the optimization strategy

The factory is responsible for:
- creating `IStudyRepository`, `IOptimizationBackendSession`, trackers, and persisters
- wiring the shared backend-study registry only when search tracking/storage is configured
- returning unentered context-manager dependencies — the orchestrator enters
  them, not the caller

## Import Rules

- Import concrete modules directly.
- Do not import from `dlkit.engine.workflows.optimization` as a barrel.
- Keep `domain` independent from `application` and `infrastructure`.
- Keep `IStudyRepository` backend-agnostic; backend-branded operations belong on
  `IOptimizationBackendSession`.
- Keep backend-study resolution in infrastructure internals; repositories do not
  expose backend-native Optuna objects as a consumed contract.

## Lifecycle Guarantees

- Backend sessions must not leave active Optuna trial handles after context exit.
- Sampling failures must finalize or discard backend trial state before re-raising.

## Entry from Runtime

```python
from dlkit.engine.workflows.entrypoints.optimization import optimize

result = optimize(settings, trials=20, study_name="search")
```
