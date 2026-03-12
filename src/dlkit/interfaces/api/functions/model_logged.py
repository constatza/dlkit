"""Run-based logged-model API helpers built on top of MLflow run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._mlflow_context import create_mlflow_client, tracking_uri_context


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggedModelRecord:
    """Run-scoped logged model metadata for search results."""

    run_id: str
    experiment_id: str
    run_name: str | None
    model_class: str | None
    artifact_path: str
    model_uri: str
    status: str | None
    start_time: int | None
    end_time: int | None
    tags: dict[str, str]


def build_logged_model_uri(
    run_id: str,
    *,
    artifact_path: str = "model",
) -> str:
    """Build canonical MLflow run-artifact model URI."""
    normalized_artifact_path = _normalize_artifact_path(artifact_path)
    return f"runs:/{run_id}/{normalized_artifact_path}"


def search_logged_models(
    model_name: str | None = None,
    *,
    experiment_name: str | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    artifact_path: str = "model",
    max_results: int = 100,
    tags: dict[str, str] | None = None,
) -> list[LoggedModelRecord]:
    """Search runs that logged a model artifact and return run-scoped model records."""
    from mlflow.entities import ViewType

    client = create_mlflow_client(tracking_uri)
    experiment_ids = _resolve_experiment_ids(
        client=client,
        experiment_name=experiment_name,
        experiment_id=experiment_id,
    )
    if not experiment_ids:
        return []

    normalized_artifact_path = _normalize_artifact_path(artifact_path)
    filter_string = _build_run_filter(
        model_name=model_name,
        run_name=run_name,
        artifact_path=normalized_artifact_path,
    )

    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["attributes.start_time DESC"],
        max_results=max_results,
    )

    required_tags = tags or {}
    return [
        _to_logged_model_record(run, normalized_artifact_path)
        for run in runs
        if _tags_match(dict(getattr(run.data, "tags", {}) or {}), required_tags)
    ]


def load_logged_model(
    *,
    run_id: str | None = None,
    model_uri: str | None = None,
    artifact_path: str = "model",
    tracking_uri: str | None = None,
) -> Any:
    """Load a model logged to a run artifact (`runs:/...`) via MLflow pyfunc."""
    import mlflow

    resolved_model_uri = _resolve_logged_model_uri(
        run_id=run_id,
        model_uri=model_uri,
        artifact_path=artifact_path,
        tracking_uri=tracking_uri,
    )
    with tracking_uri_context(tracking_uri):
        return mlflow.pyfunc.load_model(resolved_model_uri)


def _resolve_logged_model_uri(
    *,
    run_id: str | None,
    model_uri: str | None,
    artifact_path: str,
    tracking_uri: str | None,
) -> str:
    match (run_id, model_uri):
        case (str() as provided_run_id, None):
            return _get_logged_model_uri_from_run(
                run_id=provided_run_id,
                artifact_path=artifact_path,
                tracking_uri=tracking_uri,
            )
        case (None, str() as provided_model_uri):
            return provided_model_uri
        case _:
            raise ValueError("Provide exactly one of run_id or model_uri")


def _get_logged_model_uri_from_run(
    *,
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None,
) -> str:
    client = create_mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    run_tags = dict(getattr(run.data, "tags", {}) or {})
    tagged_model_uri = run_tags.get("mlflow_logged_model_uri")
    if tagged_model_uri:
        return tagged_model_uri

    return build_logged_model_uri(run_id, artifact_path=artifact_path)


def _normalize_artifact_path(artifact_path: str) -> str:
    normalized = artifact_path.strip("/")
    if not normalized:
        raise ValueError("artifact_path must not be empty")
    return normalized


def _resolve_experiment_ids(
    *,
    client: Any,
    experiment_name: str | None,
    experiment_id: str | None,
) -> list[str]:
    from mlflow.entities import ViewType

    match experiment_id:
        case str() as explicit_experiment_id if explicit_experiment_id:
            return [explicit_experiment_id]
        case _:
            pass

    match experiment_name:
        case str() as explicit_experiment_name if explicit_experiment_name:
            experiment = client.get_experiment_by_name(explicit_experiment_name)
            if experiment is None:
                return []
            return [experiment.experiment_id]
        case _:
            experiments = client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            return [experiment.experiment_id for experiment in experiments]


def _build_run_filter(
    *,
    model_name: str | None,
    run_name: str | None,
    artifact_path: str,
) -> str:
    filters = [f"tags.mlflow_logged_model_artifact_path = '{artifact_path}'"]
    if model_name:
        filters.append(f"tags.mlflow_model_class = '{model_name}'")
    if run_name:
        filters.append(f"tags.mlflow.runName = '{run_name}'")
    return " and ".join(filters)


def _tags_match(run_tags: dict[str, str], required_tags: dict[str, str]) -> bool:
    for key, expected_value in required_tags.items():
        actual_value = run_tags.get(key)
        if actual_value != expected_value:
            return False
    return True


def _to_logged_model_record(run: Any, artifact_path: str) -> LoggedModelRecord:
    run_info = run.info
    run_tags = dict(getattr(run.data, "tags", {}) or {})
    run_id = run_info.run_id
    return LoggedModelRecord(
        run_id=run_id,
        experiment_id=run_info.experiment_id,
        run_name=run_tags.get("mlflow.runName"),
        model_class=run_tags.get("mlflow_model_class"),
        artifact_path=artifact_path,
        model_uri=run_tags.get("mlflow_logged_model_uri")
        or build_logged_model_uri(run_id, artifact_path=artifact_path),
        status=str(run_info.status) if run_info.status is not None else None,
        start_time=run_info.start_time,
        end_time=run_info.end_time,
        tags=run_tags,
    )
