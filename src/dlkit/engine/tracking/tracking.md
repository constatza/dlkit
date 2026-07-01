# Runtime Tracking

`dlkit.engine.tracking` contains experiment-tracking infrastructure used by
training and optimization flows.

## Key Modules
- `interfaces.py`: tracker and run-context protocols
- `artifacts.py`: typed artifact payloads, manifests, policies, and publisher/collector protocols
- `tracking_decorator.py`: training executor decorator
- `mlflow_tracker.py`: MLflow-backed tracker
- `mlflow_run_context.py`: concrete run-context implementation
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
- Training tracking is applied through `TrackingDecorator`.
- MLflow backend selection uses `TrackingSettings.uri` when provided. Environment variables are not consulted for DLKit URI resolution.
- `TrackingDecorator` is installed only when `tracking.backend == "mlflow"` is configured.
- Training logs model artifacts under `model` and checkpoints under `checkpoints`; model registry writes are explicit public API calls, not training side effects.
- Runtime artifact publication is driven by typed `ProducedArtifact` payloads and
  a `RuntimeArtifactManifest`, not datamodule monkey-patching.
- `TrackingDecorator` computes a run-scoped `ArtifactPolicy` once and injects
  only explicit callback/output decisions downstream.
- Optimization tracker contexts are entered by runtime entrypoints/orchestrators, not by tracker factories.
- Split artifacts are logged after the run exists; generated splits are
  serialized in-memory instead of being cached to local files.
- Optimization settings/artifact manifests should prefer
  `log_artifact_content(...)` over temp-file round trips.
