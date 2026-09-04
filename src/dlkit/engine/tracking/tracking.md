# Runtime Tracking

`dlkit.engine.tracking` contains experiment-tracking infrastructure used by
training and optimization flows.

## Key Modules
- `interfaces.py`: tracker and run-context protocols
- `artifacts.py`: typed artifact payloads, manifests, policies, and publisher/collector protocols
- `tracking_decorator.py`: training executor decorator
- `lightweight_execution.py`: bare-executor training for high-volume repeated runs
  (Optuna trials) that intentionally skip full artifact logging, plus
  `fire_post_training_hooks` for firing `LifecycleHooks` on that lightweight path
- `mlflow_tracker.py`: MLflow-backed tracker
- `mlflow_resource_manager.py`: client/run lifecycle manager. Owns
  connectivity-failure handling (`TrackingSettings.on_connectivity_failure`)
  and post-fallback tag bookkeeping.
- `mlflow_client_factory.py`: client construction, connectivity probing, and
  `get_or_create_experiment` (race-tolerant experiment lookup/creation).
- `mlflow_run_context.py`: concrete run-context implementation
- `binary_artifact.py`: binary-safe temp-file staging for bytes artifacts (e.g. plot PNGs)
- `split_recovery.py`: `download_run_split()`, an explicit, user-invoked
  helper to download the `splits/*.json` artifact logged by a prior MLflow
  run into a local path (e.g. to recover a pre-fix run's split for
  evaluation, or to point `data.splits.filepath` at a run trained on a
  different machine). Never called automatically by `evaluate()` or any
  other entrypoint — split resolution fails loudly instead of silently
  invoking a recovery step on the caller's behalf.
- `run_queries.py`: `find_latest_run_id(*, experiment_name, tracking_uri=None)`
  and `find_child_run_ids(*, parent_run_id, tracking_uri=None)`, pure MLflow
  run-lookup helpers with no artifact downloading. `find_latest_run_id`
  resolves the active run with the latest `start_time` in an experiment;
  `find_child_run_ids` resolves every active run tagged
  `mlflow.parentRunId` under a given parent, ordered by ascending
  `start_time`, regardless of whether the children came from
  `MLflowResourceManager.create_run(nested=True)` or an externally tagged
  run. Both raise `WorkflowError` on a missing experiment/run or a
  zero-result search.
- `checkpoint_recovery.py`: `download_checkpoint_artifact(run_id, destination, *, tracking_uri=None)`,
  an explicit, user-invoked helper mirroring `split_recovery.py` that
  downloads a run's logged checkpoint artifact under `checkpoints/` to a
  local directory. Discovers whichever single checkpoint file actually
  exists rather than assuming a fixed filename; if more than one file is
  present it falls back to a file literally named `best.ckpt` as a
  disambiguator, and raises `WorkflowError` if none of several files is so
  named (or if the run has no checkpoint file at all). Never called
  automatically by `evaluate()` — the caller downloads the checkpoint first,
  then points the model's checkpoint override at the result.
- `backend.py`, `discovery.py`, `uri_resolver.py`: explicit backend selection and URI helpers
- `naming.py`: experiment/study naming helpers

## Notes
- Tracking scalar param maps use `ParamValue = str | int | float | bool`.
  The hook layer and the `engine.artifacts` boundary each define this same
  scalar sum type because `dlkit.engine.artifacts` is intentionally isolated in
  `tach.toml` and cannot import from `dlkit.common`.
- `interfaces.py` defines extensible tracking payload sum types for currently
  supported MLflow-facing dataset/model shapes; these are intentional sum types,
  not alias renames, and may grow as new backends are supported.
- `IRunContext` exposes `run_id`, `experiment_id`, and `tracking_uri` so result
  enrichment and artifact publication do not depend on ambient MLflow state.
- `IRunContext` uses `log_artifact_content(content, artifact_file)` for small text/bytes artifacts.
  `str` content is uploaded via MLflow's `log_text` (UTF-8 text); `bytes` content is routed through
  `binary_artifact.log_binary_artifact`, which stages it to a temp file in binary mode and uploads via
  `log_artifact` — `log_text` writes through a UTF-8 text handle and corrupts non-text bytes.
- `TrackingDecorator._is_tracking_owner(components)` (a shared static helper)
  is checked by **both** `execute()` and `execute_within_run()` — skipping
  tracking entirely on any process where `components.trainer.is_global_zero`
  is `False` (delegating straight to the wrapped executor instead), and
  treating components with no trainer at all as always the owner. Under
  multi-process execution (DDP, whether via local multi-GPU, SLURM,
  torchrun, LSF, or MPI — see `dlkit.infrastructure.compute.compute.md`),
  Lightning re-executes or independently launches the entire process per
  rank, so every rank would otherwise call `mlflow.create_run`/`start_run`
  redundantly (via `execute()`), or write conflicting metadata/callbacks/
  metrics into the same already-open run (via `execute_within_run()` — the
  HPO best-retrain/multirun-child path, where every rank shares one
  caller-opened `run_context`, so double writes there are a correctness bug,
  not just a duplicate run). Only rank 0 owns the MLflow run lifecycle
  either way.
- Training tracking is applied through `TrackingDecorator`. `execute()` owns opening
  the MLflow run; `execute_within_run(components, settings, *, run_context,
  tracking_uri=None)` runs the identical logging/callback/hook pipeline against a run
  the caller already opened, and never fires `on_run_created` (the caller does, since
  it already has the run_id) or calls `create_run` itself. This is how search/HPO's
  best-retrain and `MultiRunOrchestrator`'s sweep children get full `train()`-parity
  artifact logging without opening a second, wrongly-nested run — the caller must pass
  the same tracker instance that opened `run_context`, since nested-run detection in
  `MLflowResourceManager` is per-instance, not global.
- MLflow backend selection uses `TrackingSettings.uri` when provided. Environment variables are not consulted for DLKit URI resolution.
- `TrackingDecorator` is installed only when `tracking.backend == "mlflow"` is configured.
- Training logs deployment model artifacts under `model` and Lightning checkpoints
  under `checkpoints`; `.ckpt` files remain internal training artifacts for
  resume/debug/best-last history and are not reused to build the MLflow model
  artifact.
- PyTorch model artifacts use `TrackingSettings.model_serialization_format`.
  `"pickle"` preserves legacy MLflow behavior; `"pt2"` opts into MLflow's
  `torch.export`-backed serialization and requires input-shape metadata so an
  `input_example` can be built.
- `artifact_logger._build_pt2_signature` only names `TensorSpec` inputs when a
  model has more than one input. MLflow's pytorch flavor pyfunc wrapper only
  accepts a bare ndarray/DataFrame at predict time; a *named* single-input
  schema forces MLflow's own schema enforcement to wrap the example in a dict,
  which the wrapper rejects (`mlflow.pyfunc.load_model(uri).predict(...)`
  fails, and `log_model` emits "Failed to validate serving input example").
  Multi-input models still need names to disambiguate shapes, but MLflow's
  pytorch flavor has no working pyfunc/REST-serving path for multi-tensor
  inputs regardless of naming — `_log_model_artifact` logs an explicit warning
  for that case since MLflow itself fails silently (dropped input example
  under `"pickle"`, missing `python_function` flavor under `"pt2"`).
- Model registry writes are explicit public API calls, not training side effects.
- Runtime artifact publication is driven by typed `ProducedArtifact` payloads and
  a `RuntimeArtifactManifest`, not datamodule monkey-patching.
- `TrackingDecorator` computes a run-scoped `ArtifactPolicy` once and injects
  only explicit callback/output decisions downstream.
- Optimization tracker contexts are entered by runtime entrypoints/orchestrators, not by tracker factories.
- `MLflowResourceManager.reset_global_state()` drains every open MLflow run
  (`while mlflow.active_run(): mlflow.end_run()`, then a hard clear of
  `mlflow.tracking.fluent._active_run_stack`) rather than assuming a fixed
  number of open runs. This runs on every `MLflowResourceManager.__exit__`
  and is what lets sequential `train()`/`optimize()` calls share a process
  without a later call inheriting an earlier one's still-active run.
- `ArtifactLogger.log_checkpoints` has two independent sources: a live
  `Trainer`'s `ModelCheckpoint` callback (the gradient-training path), and any
  `ProducedArtifact`s attached directly to `components.model` via
  `engine.artifacts.attach_checkpoint_artifacts`/`read_checkpoint_artifacts`
  (the trainer-free one-shot fit path — see
  `engine.training.execution.md`'s `OneShotFitExecutor` section).
  The model-attached channel exists because `TrackingDecorator` snapshots
  `RuntimeComponents` *before* `ITrainingExecutor.execute()` runs, so a
  checkpoint produced during `execute()` can't reach `ArtifactLogger` through
  `RuntimeComponents.artifacts` — the live model object, shared by reference
  across every snapshot, is the only channel that does.
- Split artifacts are logged after the run exists. Generated splits are
  additionally persisted to a local `splits/` file under
  `training.trainer.default_root_dir` when that root is configured (see
  `dlkit.infrastructure.io.io.md`), independent of MLflow artifact logging.
- `IExperimentTracker.is_active()` (default `True`; `NullTracker` overrides
  to `False`) is the capability query `TrackingDecorator` uses to compute
  `tracking_enabled` for `ArtifactPolicy`, instead of a hardcoded `True`.
  Combined with `ArtifactPolicy.remove_uploaded_files` actually being read
  by `ArtifactLogger._log_or_skip_checkpoint` before deleting a local
  checkpoint (previously computed but never consulted), a fully local or
  untracked run's checkpoint is never deleted, and a real upload failure
  correctly prevents deletion rather than being silently swallowed.
- Optimization settings/artifact manifests should prefer
  `log_artifact_content(...)` over temp-file round trips.
- Every `ClientBasedRunContext` method except `log_model` is wrapped in
  `@best_effort`, which catches all exceptions and logs a warning instead of
  raising. `log_model` is the one exception — it raises, since callers must
  not silently proceed without the saved model artifact. MLflow's HTTP
  retry/timeout/backoff are process-global env vars
  (`infrastructure.config.environment.ensure_mlflow_defaults`), so
  `best_effort` scopes them down to a fail-fast budget
  (`best_effort_retry_budget`) for the duration of each wrapped call — a call
  that's allowed to fail shouldn't burn the wider budget sized for
  `log_model`. `TrackingSettings.max_retries` still governs `log_model`'s
  budget (and the process default) untouched.
- `MLflowTracker.set_run_tag(run_id, key, value)` and `.get_run_context(run_id)` are
  non-activating, client-backed (`MlflowClient.set_tag(...)` / a read-only
  `ClientBasedRunContext`) — neither calls `mlflow.start_run()`, so both are safe to use
  on a run that isn't the tracker's currently-active one. `MultiRunOrchestrator` uses
  `set_run_tag` to tag a heterogeneous sweep child's already-closed, independently-opened
  run with `mlflow.parentRunId` after the fact (each child workflow — `train`/`optimize`/
  `converge` — opens and closes its own top-level run via `engine.workflows.entrypoints.execute()`;
  holding the sweep's parent run open across all of them is not possible since
  `mlflow.start_run()` is process-global, not per-tracker-instance, and two concurrently
  "active" runs raise). The same mechanism tags the parent run
  `"multirun.status" = "failed"` under `FailurePolicy = "continue_mark_parent_failed"`
  once any child fails. See `dlkit.engine.workflows.entrypoints.entrypoints.md` for the
  sweep orchestration this supports.
- `MLflowResourceManager._initialize_resources` treats a failed connectivity
  probe as fatal by default (`TrackingSettings.on_connectivity_failure =
  "raise"`, raising `TrackingError`) rather than logging a warning and
  continuing against an unreachable backend — the latter used to silently
  fall through to MLflow's own multi-minute default HTTP retry/backoff on
  every subsequent call, which looked like a hang, not an error. Setting
  `on_connectivity_failure = "fallback_local"` explicitly opts into tracking
  locally instead (via `select_backend(probe=lambda: False)`, never the
  original unreachable `uri`); the resulting run is tagged
  `tracking.degraded_fallback`/`tracking.degraded_reason` so it's never
  mistaken for one that reached the configured backend. A backend that's
  already `LocalSqliteBackend` and unreachable always raises — there's no
  meaningful fallback target from local. `MLflowTracker.get_tracking_uri()`/
  `is_local()` delegate to `MLflowResourceManager`'s live `backend` property
  rather than a snapshot taken at `__enter__`, so they reflect a mid-init
  fallback correctly.
- `MLflowClientFactory.get_or_create_experiment` tolerates losing a create
  race against a concurrent creator (e.g. two processes pointed at the same
  local SQLite backend): a `create_experiment` call that fails with
  `MlflowException(error_code="RESOURCE_ALREADY_EXISTS")` re-fetches the
  experiment by name instead of propagating the raw SQLAlchemy
  `IntegrityError`-derived exception. Any other `MlflowException` still
  propagates unchanged.
- `raise_error` (`infrastructure.utils.error_handling`) re-raises a
  `DLKitError` passed as `original_error` unchanged instead of wrapping it in
  `error_class` — this is what lets a `TrackingError` (or any other typed
  error) survive the several `except Exception: raise_error(...)` boundaries
  between `with tracker:` and the CLI without being flattened into a generic
  `WorkflowError`.
- Metric stage identifiers (`dlkit.common.metric_stages.MetricStage`) flow as
  the enum end-to-end, from the Lightning wrapper's step logger through
  `MLflowEpochLogger` and `MetricLogger` — there is no string-typed stage
  parameter anywhere in this path. This closes off a past bug class where a
  malformed stage string (e.g. `"val_epoch"`) silently produced a bogus
  `val_epoch/*` MLflow metric group instead of failing loudly; a caller
  passing anything other than a `MetricStage` member is now a type error, not
  a silent runtime fallback. Runs logged before this fix may still show
  `val_epoch/`/`test_epoch/` groups in the MLflow UI — that's permanent
  history on those runs, not a live bug; `scripts/purge_legacy_epoch_metric_runs.py`
  finds (and, with `--delete`, removes) affected runs.
