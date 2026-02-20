"""Model registry API helpers built on top of MLflow primitives."""

from __future__ import annotations

from typing import Any, Literal

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


def register_logged_model(
    model_name: str,
    *,
    run_id: str,
    artifact_path: str = "model",
    tracking_uri: str | None = None,
) -> Any:
    """Register a run-logged model artifact as a model version.

    Creates the registered model if it does not exist.
    """
    client = create_mlflow_client(tracking_uri)
    _ensure_registered_model_exists(client, model_name)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )


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
                return mlflow.pytorch.load_model(model_uri)  # type: ignore[attr-defined]
            case "sklearn":
                return mlflow.sklearn.load_model(model_uri)  # type: ignore[attr-defined]
            case "pyfunc":
                return mlflow.pyfunc.load_model(model_uri)
            case "auto":
                for loader in (
                    mlflow.pytorch.load_model,  # type: ignore[attr-defined]
                    mlflow.sklearn.load_model,  # type: ignore[attr-defined]
                    mlflow.pyfunc.load_model,
                ):
                    try:
                        return loader(model_uri)
                    except Exception:
                        continue
                raise RuntimeError(f"Failed to load model URI '{model_uri}' with all supported loaders")
            case _:
                raise ValueError(f"Unsupported flavor strategy '{flavor}'")


def _ensure_registered_model_exists(client: Any, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message:
            return
        raise
