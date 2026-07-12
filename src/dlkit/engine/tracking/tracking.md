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
- `mlflow_run_context.py`: concrete run-context implementation
- `binary_artifact.py`: binary-safe temp-file staging for bytes artifacts (e.g. plot PNGs)
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
- Split artifacts are logged after the run exists; generated splits are
  serialized in-memory instead of being cached to local files.
- Optimization settings/artifact manifests should prefer
  `log_artifact_content(...)` over temp-file round trips.
