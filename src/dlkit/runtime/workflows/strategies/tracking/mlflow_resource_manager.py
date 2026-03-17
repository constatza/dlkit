"""MLflow resource manager for client-only lifecycle management."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import threading

import os

import mlflow
from mlflow import MlflowClient

from dlkit.runtime.workflows.strategies.tracking.uri_resolver import (
    parse_mlflow_scheme,
    resolve_mlflow_uris,
)
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.io import url_resolver
from dlkit.tools.utils.logging_config import get_logger

from .mlflow_client_factory import MLflowClientFactory

logger = get_logger(__name__)


@dataclass(slots=True)
class MLflowResourceState:
    """Mutable resource lifecycle state for MLflow manager internals."""

    client: MlflowClient | None = None
    active_run_stack: list[str] = field(default_factory=list)
    stack_lock: threading.Lock = field(default_factory=threading.Lock)
    experiment_id: str | None = None
    cleanup_callbacks: list[Any] = field(default_factory=list)
    tracking_uri: str | None = None
    artifact_uri: str | None = None


class MLflowResourceManager:
    """Centralized MLflow client/run resource manager."""

    def __init__(self, mlflow_config: MLflowSettings | None = None):
        self._config = mlflow_config
        self._state = MLflowResourceState()
        self._is_initialized = False

    def __enter__(self) -> MLflowResourceManager:
        if self._is_initialized:
            raise RuntimeError("Resource manager already initialized")
        self._initialize_resources()
        self._is_initialized = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup_all_resources()
        self._is_initialized = False

    def _initialize_resources(self) -> None:
        from dlkit.tools.config.environment import ensure_mlflow_defaults

        ensure_mlflow_defaults()

        resolved = resolve_mlflow_uris()
        self._state.tracking_uri = resolved.tracking_uri
        self._state.artifact_uri = resolved.artifact_uri

        self._ensure_local_storage_if_needed(resolved.tracking_uri, resolved.artifact_uri)
        self._set_global_tracking_uri(resolved.tracking_uri)

        self._state.client = MLflowClientFactory.create_client(tracking_uri=resolved.tracking_uri)
        if not MLflowClientFactory.validate_client_connectivity(self._state.client):
            logger.warning("MLflow client connectivity validation failed")

    def _ensure_local_storage_if_needed(self, tracking_uri: str, artifact_uri: str | None) -> None:
        match parse_mlflow_scheme(tracking_uri):
            case "sqlite":
                db_path = self._sqlite_db_path(tracking_uri)
                db_path.parent.mkdir(parents=True, exist_ok=True)
            case "http" | "https":
                pass
            case unexpected:
                raise ValueError(f"Unsupported tracking scheme: {unexpected}")

        if artifact_uri and artifact_uri.startswith("file://"):
            artifact_path = url_resolver.resolve_local_uri(artifact_uri, Path.cwd())
            artifact_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sqlite_db_path(uri: str) -> Path:
        return url_resolver.resolve_local_uri(uri, Path.cwd())

    def _set_global_tracking_uri(self, uri: str) -> None:
        current_uri = self._state.tracking_uri
        if current_uri and current_uri != uri:
            msg = (
                f"Attempting to change global tracking URI from {current_uri} to {uri}. "
                "This indicates inconsistent MLflow resource state."
            )
            raise RuntimeError(msg)
        mlflow.set_tracking_uri(uri)
        self._state.tracking_uri = uri

    def get_client(self) -> MlflowClient:
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized - use as context manager")
        if self._state.client is None:
            raise RuntimeError("MLflow client not available")
        return self._state.client

    def get_tracking_uri(self) -> str | None:
        """Get currently resolved tracking URI."""
        return self._state.tracking_uri

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ):
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized")

        self._validate_stack_consistency()
        client = self.get_client()
        exp_name = experiment_name or "DLKit"
        experiment_id = MLflowClientFactory.get_or_create_experiment(
            client,
            exp_name,
            self._state.artifact_uri,
        )
        self._state.experiment_id = experiment_id

        parent_run_id = None
        with self._state.stack_lock:
            if nested and self._state.active_run_stack:
                parent_run_id = self._state.active_run_stack[-1]

        start_kwargs: dict[str, Any] = {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "nested": nested,
        }
        if parent_run_id:
            start_kwargs["parent_run_id"] = parent_run_id
        if tags:
            start_kwargs["tags"] = tags

        with mlflow.start_run(**start_kwargs) as active_run:
            run_id = active_run.info.run_id
            with self._state.stack_lock:
                self._state.active_run_stack.append(run_id)

            try:
                from .mlflow_run_context import ClientBasedRunContext

                yield ClientBasedRunContext(client, run_id)
            finally:
                with self._state.stack_lock:
                    if self._state.active_run_stack and self._state.active_run_stack[-1] == run_id:
                        self._state.active_run_stack.pop()

        self._validate_stack_consistency()

    def add_cleanup_callback(self, callback: Any) -> None:
        self._state.cleanup_callbacks.append(callback)

    def _validate_stack_consistency(self) -> None:
        global_active = mlflow.active_run()
        with self._state.stack_lock:
            match (bool(self._state.active_run_stack), bool(global_active)):
                case (False, False):
                    return
                case (True, True):
                    internal_active = self._state.active_run_stack[-1]
                    global_run_id = global_active.info.run_id
                    if internal_active != global_run_id:
                        logger.warning(
                            f"Stack desynchronization: internal={internal_active} "
                            f"global={global_run_id}"
                        )
                case _:
                    logger.warning(
                        "Stack desynchronization: "
                        f"internal_depth={len(self._state.active_run_stack)} "
                        f"global_present={bool(global_active)}"
                    )

    def _get_state_snapshot(self) -> dict[str, Any]:
        """Get current state snapshot for debugging and tests."""
        with self._state.stack_lock:
            global_run = mlflow.active_run()
            return {
                "initialized": self._is_initialized,
                "tracking_uri": self._state.tracking_uri,
                "client_exists": self._state.client is not None,
                "active_run_stack": list(self._state.active_run_stack),
                "stack_depth": len(self._state.active_run_stack),
                "experiment_id": self._state.experiment_id,
                "mlflow_global_run": global_run.info.run_id if global_run else None,
            }

    def _cleanup_all_resources(self) -> None:
        with self._state.stack_lock:
            if self._state.active_run_stack and self._state.client:
                runs_to_cleanup = list(reversed(self._state.active_run_stack))
                for run_id in runs_to_cleanup:
                    try:
                        self._state.client.set_terminated(run_id, status="FINISHED")
                    except Exception as e:  # pragma: no cover - best-effort safety
                        logger.warning("Failed to terminate active run {}: {}", run_id, e)
                self._state.active_run_stack.clear()

        try:
            mlflow.end_run()
        except Exception:
            pass

        for callback in reversed(self._state.cleanup_callbacks):
            try:
                callback()
            except Exception as e:  # pragma: no cover - best-effort safety
                logger.warning("Cleanup callback failed: {}", e)

        self.reset_global_state()
        self._state = MLflowResourceState()

    @staticmethod
    def reset_global_state() -> None:
        try:
            try:
                mlflow.end_run()
            except Exception:
                pass
            # set_tracking_uri(None) in MLflow 3.x removes MLFLOW_TRACKING_URI from
            # os.environ when mlflow previously set it.  Preserve the env var so that
            # any code running after this reset still uses the correct tracking backend.
            _saved_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(None)  # type: ignore[arg-type]
            if _saved_uri is not None:
                os.environ["MLFLOW_TRACKING_URI"] = _saved_uri
            if hasattr(mlflow, "_active_run_stack"):
                mlflow._active_run_stack.clear()  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover - best-effort safety
            logger.warning("Failed to reset MLflow global state: {}", e)
