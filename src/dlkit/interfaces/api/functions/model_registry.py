"""Model registry API helpers built on top of MLflow primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mlflow.exceptions import MlflowException

from dlkit.engine.tracking.artifact_logger import (
    CHECKPOINT_ARTIFACT_DIR,
    DEFAULT_MODEL_ARTIFACT_PATH,
    TAG_LOGGED_MODEL_URI,
)
from dlkit.engine.tracking.checkpoint_recovery import (
    download_checkpoint_artifact as _download_checkpoint_artifact,
)
from dlkit.engine.tracking.run_queries import find_child_run_ids as _find_child_run_ids
from dlkit.engine.tracking.run_queries import find_latest_run_id as _find_latest_run_id
from dlkit.engine.tracking.split_recovery import download_run_split as _download_run_split

from ._mlflow_context import create_mlflow_client, tracking_uri_context


def search_registered_models(
    model_name: str | None = None,
    *,
    tracking_uri: str | None = None,
) -> list[Any]:
    """Search registered models, optionally filtering by exact model name."""
    client = create_mlflow_client(tracking_uri)
    if not model_name:
        return list(client.search_registered_models())
    return list(client.search_registered_models(filter_string=f"name = '{model_name}'"))


def list_model_versions(
    model_name: str,
    *,
    tracking_uri: str | None = None,
) -> list[int]:
    """Return all available versions for a registered model."""
    client = create_mlflow_client(tracking_uri)
    versions = client.search_model_versions(f"name = '{model_name}'")
    return sorted(int(v.version) for v in versions)


def get_model_version(
    model_name: str,
    version: int,
    *,
    tracking_uri: str | None = None,
) -> Any:
    """Get a specific registered model version entity."""
    client = create_mlflow_client(tracking_uri)
    return client.get_model_version(name=model_name, version=str(version))


def has_checkpoint_artifact(
    run_id: str,
    *,
    tracking_uri: str | None = None,
) -> bool:
    """Return True if the run has at least one checkpoint artifact logged.

    Args:
        run_id: MLflow run identifier.
        tracking_uri: Optional tracking URI (uses active MLflow config when omitted).

    Returns:
        True when the run's ``checkpoints/`` artifact directory is non-empty.
    """
    client = create_mlflow_client(tracking_uri)
    try:
        artifacts = client.list_artifacts(run_id, path=CHECKPOINT_ARTIFACT_DIR)
        return bool(artifacts)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise


def download_run_split(
    run_id: str,
    destination: Path,
    *,
    tracking_uri: str | None = None,
) -> Path:
    """Download a previously trained run's persisted split for recovery.

    Standalone recovery utility — not invoked automatically by ``evaluate()``.
    Point ``data.splits.filepath`` at the returned path to reuse an old run's
    exact held-out split rather than an unrelated regenerated one.

    Args:
        run_id: MLflow run ID that logged a ``splits/*.json`` artifact.
        destination: Local directory to download the split file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded split file.
    """
    return _download_run_split(run_id, destination, tracking_uri=tracking_uri)


def find_latest_run_id(
    *,
    experiment_name: str,
    tracking_uri: str | None = None,
) -> str:
    """Find the most recently started active run in an experiment.

    Standalone recovery utility for run-based checkpoint selection — not
    invoked automatically by ``evaluate()``. Use the returned run id with
    ``download_checkpoint_artifact()`` or a ``RunCheckpoint`` override.

    Args:
        experiment_name: Name of the MLflow experiment to search.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Run id of the active run with the latest ``start_time`` in the
        experiment.
    """
    return _find_latest_run_id(experiment_name=experiment_name, tracking_uri=tracking_uri)


def find_child_run_ids(
    *,
    parent_run_id: str,
    tracking_uri: str | None = None,
) -> tuple[str, ...]:
    """Find all active child runs of a parent run, in creation order.

    Standalone recovery utility for run-based checkpoint selection — not
    invoked automatically by ``evaluate()``.

    Args:
        parent_run_id: MLflow run id of the parent run.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Tuple of child run ids ordered by ascending ``start_time``.
    """
    return _find_child_run_ids(parent_run_id=parent_run_id, tracking_uri=tracking_uri)


def download_checkpoint_artifact(
    run_id: str,
    destination: Path,
    *,
    tracking_uri: str | None = None,
) -> Path:
    """Download a run's logged checkpoint artifact to a local directory.

    Standalone recovery utility — not invoked automatically by
    ``evaluate()``. Point the model's checkpoint override at the returned
    path to reuse an old run's exact checkpoint rather than an unrelated one.

    Args:
        run_id: MLflow run id that logged a checkpoint artifact during
            training.
        destination: Local directory to download the checkpoint file into.
        tracking_uri: Optional explicit MLflow tracking URI override.

    Returns:
        Path to the downloaded checkpoint file.
    """
    return _download_checkpoint_artifact(run_id, destination, tracking_uri=tracking_uri)


def register_logged_model(
    model_name: str,
    *,
    run_id: str,
    artifact_path: str = DEFAULT_MODEL_ARTIFACT_PATH,
    tracking_uri: str | None = None,
) -> Any:
    """Register a run-logged model artifact as a model version.

    Creates the registered model if it does not exist.
    """
    client = create_mlflow_client(tracking_uri)
    run = client.get_run(run_id)
    tags = getattr(getattr(run, "data", None), "tags", {})
    tagged_uri = tags.get(TAG_LOGGED_MODEL_URI) if isinstance(tags, dict) else None
    model_uri = tagged_uri or f"runs:/{run_id}/{artifact_path}"

    import mlflow

    with tracking_uri_context(tracking_uri):
        return mlflow.register_model(model_uri=model_uri, name=model_name)


def set_registered_model_alias(
    model_name: str,
    *,
    alias: str,
    version: int,
    tracking_uri: str | None = None,
) -> None:
    """Set a registered-model alias to a specific version."""
    client = create_mlflow_client(tracking_uri)
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=str(version),
    )


def set_registered_model_version_tag(
    model_name: str,
    *,
    version: int,
    key: str,
    value: str,
    tracking_uri: str | None = None,
) -> None:
    """Set a single model-version tag."""
    client = create_mlflow_client(tracking_uri)
    client.set_model_version_tag(
        name=model_name,
        version=str(version),
        key=key,
        value=value,
    )


def set_registered_model_version_tags(
    model_name: str,
    *,
    version: int,
    tags: dict[str, str],
    tracking_uri: str | None = None,
) -> None:
    """Set multiple model-version tags."""
    for key, value in tags.items():
        set_registered_model_version_tag(
            model_name,
            version=version,
            key=key,
            value=value,
            tracking_uri=tracking_uri,
        )


def build_registered_model_uri(
    model_name: str,
    *,
    version: int | None = None,
    alias: str | None = None,
) -> str:
    """Build a canonical MLflow model URI for version or alias selection."""
    if version is not None and alias is not None:
        raise ValueError("Provide either version or alias, not both")

    match version:
        case int() as explicit_version:
            return f"models:/{model_name}/{explicit_version}"
        case _:
            resolved_alias = alias or "latest"
            return f"models:/{model_name}@{resolved_alias}"


def load_registered_model(
    model_name: str,
    *,
    version: int | None = None,
    alias: str | None = None,
    tracking_uri: str | None = None,
    flavor: Literal["auto", "pytorch", "sklearn", "pyfunc"] = "auto",
) -> Any:
    """Load model from MLflow registry using version or alias.

    Alias URIs (``models:/<name>@<alias>``) are fully supported.
    """
    import mlflow

    model_uri = build_registered_model_uri(model_name, version=version, alias=alias)
    with tracking_uri_context(tracking_uri):
        match flavor:
            case "pytorch":
                return mlflow.pytorch.load_model(model_uri)
            case "sklearn":
                return mlflow.sklearn.load_model(model_uri)
            case "pyfunc":
                return mlflow.pyfunc.load_model(model_uri)
            case "auto":
                errors: dict[str, Exception] = {}
                for loader in (
                    mlflow.pytorch.load_model,
                    mlflow.sklearn.load_model,
                    mlflow.pyfunc.load_model,
                ):
                    try:
                        return loader(model_uri)
                    except Exception as e:  # noqa: BLE001 - collected below, not swallowed
                        errors[loader.__module__] = e
                last_exc = next(reversed(errors.values()))
                summary = "; ".join(f"{module}: {exc}" for module, exc in errors.items())
                raise RuntimeError(
                    f"Failed to load model URI '{model_uri}' with all supported loaders ({summary})"
                ) from last_exc
            case _:
                raise ValueError(f"Unsupported flavor strategy '{flavor}'")
