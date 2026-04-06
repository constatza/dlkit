# Runtime Tracking

`dlkit.engine.tracking` contains experiment-tracking infrastructure used by
training and optimization flows.

## Key Modules
- `interfaces.py`: tracker and run-context protocols
- `tracking_decorator.py`: training executor decorator
- `mlflow_tracker.py`: MLflow-backed tracker
- `mlflow_run_context.py`: concrete run-context implementation
- `backend.py`, `discovery.py`, `uri_resolver.py`: backend selection and probing
- `naming.py`: experiment/study naming helpers

## Notes
- `IRunContext` uses `log_artifact_content(content, artifact_file)` for small text/bytes artifacts.
- Training tracking is applied through `TrackingDecorator`.
- Optimization tracker contexts are entered by runtime entrypoints/orchestrators, not by tracker factories.
