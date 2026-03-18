"""MLflow resource manager for client-only lifecycle management."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import threading

import mlflow
from mlflow import MlflowClient

from dlkit.runtime.workflows.strategies.tracking.backend import (
    LocalSqliteBackend,
    RemoteServerBackend,
    TrackingBackend,
)
from dlkit.runtime.workflows.strategies.tracking.uri_resolver import parse_mlflow_scheme
from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.io import url_resolver
from dlkit.tools.utils.logging_config import (
    get_logger,
    suppress_mlflow_sqlite_bootstrap_logs,
)

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


class MLflowResourceManager:
    """Centralized MLflow client/run resource manager."""

    def __init__(self, mlflow_config: MLflowSettings | None, backend: TrackingBackend):
        """Initialize with MLflow config and pre-selected backend.

        Args:
            mlflow_config: Optional MLflow settings.
            backend: Pre-selected tracking backend (from select_backend()).
        """
        self._config = mlflow_config
        self._backend = backend
        self._state = MLflowResourceState()
        self._is_initialized = False

    def __enter__(self) -> MLflowResourceManager:
        if self._is_initialized:
            raise RuntimeError("Resource manager already initialized")
        self._initialize_resources()
        self._is_initialized = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        with self._state.stack_lock:
            if self._state.active_run_stack:
                logger.warning("Non-empty run stack at cleanup: {}", self._state.active_run_stack)
        self._cleanup_all_resources()
        self._is_initialized = False

    def _initialize_resources(self) -> None:
        from dlkit.tools.config.environment import ensure_mlflow_defaults

        ensure_mlflow_defaults()

        tracking_uri = self._backend.tracking_uri()
        artifact_uri = self._backend.artifact_uri()

        self._ensure_local_storage_if_needed(artifact_uri)

        with self._sqlite_bootstrap_log_suppressed(self._backend.scheme()):
            self._state.client = MLflowClientFactory.create_client(
                tracking_uri=tracking_uri
            )
            if not MLflowClientFactory.validate_client_connectivity(self._state.client):
                logger.warning("MLflow client connectivity validation failed")

    def _ensure_local_storage_if_needed(self, artifact_uri: str | None) -> None:
        match self._backend:
            case LocalSqliteBackend(db_path=db_path):
                db_path.parent.mkdir(parents=True, exist_ok=True)
            case _:
                pass

        if artifact_uri and artifact_uri.startswith("file://"):
            artifact_path = url_resolver.resolve_local_uri(artifact_uri, Path.cwd())
            artifact_path.mkdir(parents=True, exist_ok=True)

    def get_client(self) -> MlflowClient:
        """Return the initialized MLflow client.

        Returns:
            Active MlflowClient instance.

        Raises:
            RuntimeError: If not initialized or client unavailable.
        """
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized - use as context manager")
        if self._state.client is None:
            raise RuntimeError("MLflow client not available")
        return self._state.client

    def get_tracking_uri(self) -> str | None:
        """Return the backend's tracking URI.

        Returns:
            Tracking URI string or None if not initialized.
        """
        if not self._is_initialized:
            return None
        return self._backend.tracking_uri()

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ):
        """Create a scoped MLflow run context.

        Sets tracking URI for the duration of the run only and restores it
        on exit. For SQLite backends the URI is cleared; for HTTP backends
        the user-configured URI is preserved.

        Args:
            experiment_name: Experiment to associate the run with.
            run_name: Optional run name.
            nested: If True, creates a child run under the current parent.
            tags: Optional tags to attach.

        Yields:
            ClientBasedRunContext: Active run context.

        Raises:
            RuntimeError: If not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized")

        client = self.get_client()
        exp_name = experiment_name or "DLKit"
        tracking_uri = self._backend.tracking_uri()
        artifact_uri = self._backend.artifact_uri()

        with self._sqlite_bootstrap_log_suppressed(self._backend.scheme()):
            experiment_id = MLflowClientFactory.get_or_create_experiment(
                client,
                exp_name,
                artifact_uri,
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

        # Scoped tracking URI — set only for the duration of this run
        mlflow.set_tracking_uri(tracking_uri)
        try:
            with self._sqlite_bootstrap_log_suppressed(self._backend.scheme()):
                with mlflow.start_run(**start_kwargs) as active_run:
                    run_id = active_run.info.run_id
                    with self._state.stack_lock:
                        self._state.active_run_stack.append(run_id)

                    try:
                        from .mlflow_run_context import ClientBasedRunContext

                        yield ClientBasedRunContext(client, run_id, tracking_uri=tracking_uri)
                    finally:
                        with self._state.stack_lock:
                            if self._state.active_run_stack and self._state.active_run_stack[-1] == run_id:
                                self._state.active_run_stack.pop()
        finally:
            self._restore_tracking_uri_if_last_run()

    def _restore_tracking_uri_if_last_run(self) -> None:
        """Restore or clear tracking URI only when the outermost run exits.

        Prevents nested runs from clearing the URI while a parent run is still
        active and needs it to communicate with the MLflow store.

        Note:
            In MLflow 3.x, ``mlflow.set_tracking_uri(None)`` does **not** fall
            back to the ``MLFLOW_TRACKING_URI`` environment variable — it resets
            the internal state to the CWD-relative default (``sqlite:///mlflow.db``),
            which would create a stray DB in the project root.  For non-remote
            backends we therefore call ``mlflow.set_tracking_uri(env_uri)`` with
            the current env-var value so all subsequent ``get_tracking_uri()``
            calls remain pointed at the correct isolated path.
        """
        with self._state.stack_lock:
            still_active = bool(self._state.active_run_stack)
        if still_active:
            return
        match self._backend:
            case RemoteServerBackend(uri=uri):
                os.environ["MLFLOW_TRACKING_URI"] = uri
                mlflow.set_tracking_uri(uri)
            case _:
                # In MLflow 3.x set_tracking_uri(None) ignores the env var and
                # defaults to the CWD-relative mlflow.db. Use the env var value
                # so subsequent mlflow.get_tracking_uri() calls stay isolated.
                env_uri = os.environ.get("MLFLOW_TRACKING_URI")
                mlflow.set_tracking_uri(env_uri)  # type: ignore[arg-type]

    def add_cleanup_callback(self, callback: Any) -> None:
        """Register a cleanup callback to run on context exit.

        Args:
            callback: Zero-argument callable invoked during cleanup.
        """
        self._state.cleanup_callbacks.append(callback)

    @staticmethod
    def _sqlite_bootstrap_log_suppressed(scheme: str):
        if scheme == "sqlite":
            return suppress_mlflow_sqlite_bootstrap_logs()
        return nullcontext()

    def _get_state_snapshot(self) -> dict[str, Any]:
        """Return current state snapshot for debugging and tests."""
        with self._state.stack_lock:
            global_run = mlflow.active_run()
            return {
                "initialized": self._is_initialized,
                "tracking_uri": self._backend.tracking_uri() if self._is_initialized else None,
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
        """Reset MLflow global state.

        Ends any active runs, clears the run stack, and resets the tracking URI
        to the current ``MLFLOW_TRACKING_URI`` env-var value (if set).

        Note:
            In MLflow 3.x, ``mlflow.set_tracking_uri(None)`` does **not** fall
            back to the ``MLFLOW_TRACKING_URI`` environment variable — it resets
            the internal state to the CWD-relative default (``sqlite:///mlflow.db``),
            which would create a stray DB in the project root.  We therefore
            re-apply the env-var URI explicitly so any code running after this
            call (e.g. fixture teardowns, ``autolog``) continues using the
            correct isolated path.
        """
        try:
            try:
                mlflow.end_run()
            except Exception:
                pass
            if hasattr(mlflow, "_active_run_stack"):
                mlflow._active_run_stack.clear()  # type: ignore[attr-defined]
            # In MLflow 3.x, set_tracking_uri(None) ignores the env var and
            # defaults to the CWD-relative mlflow.db. Use the env var value so
            # the isolation URI set by fixtures is preserved after reset.
            saved = os.environ.get("MLFLOW_TRACKING_URI")
            if saved:
                mlflow.set_tracking_uri(saved)
                os.environ["MLFLOW_TRACKING_URI"] = saved
            else:
                mlflow.set_tracking_uri(None)  # type: ignore[arg-type]
        except Exception as e:  # pragma: no cover - best-effort safety
            logger.warning("Failed to reset MLflow global state: {}", e)
