"""MLflow resource manager for centralized resource lifecycle management."""

from __future__ import annotations

from typing import Any
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading

import mlflow
from mlflow import MlflowClient

from dlkit.tools.config.mlflow_settings import MLflowSettings
from dlkit.tools.utils.logging_config import get_logger
from dlkit.interfaces.servers import health_checker as health_checker_module
from dlkit.interfaces.servers import mlflow_adapter as mlflow_adapter_module

from .mlflow_client_factory import MLflowClientFactory
from dlkit.tools.io.path_normalizers import canonicalize_file_uri


# Sentinels allow tests to override server primitives while enabling integration
# tests to patch the adapter module directly.
_USE_MODULE_CONTEXT = object()
_USE_MODULE_ADAPTER = object()


# Public attributes that tests can patch directly. When left untouched we fall
# back to the implementations exposed by ``mlflow_adapter_module``.
MLflowServerContext = _USE_MODULE_CONTEXT  # type: ignore[assignment]
MLflowServerAdapter = _USE_MODULE_ADAPTER  # type: ignore[assignment]

logger = get_logger(__name__)


def _normalize_tracking_uri(uri: str | None) -> str | None:
    """Normalize tracking URI for consistent comparison.

    Strips trailing slashes to ensure URLs like 'http://host:port' and
    'http://host:port/' are treated as equivalent.

    Args:
        uri: Tracking URI to normalize

    Returns:
        Normalized URI with trailing slashes removed, or None if input is None
    """
    if not uri:
        return uri

    stripped = uri.rstrip("/")

    if stripped.lower().startswith("file://"):
        return canonicalize_file_uri(stripped)

    return stripped


def _resolve_server_context_cls():
    override = globals().get("MLflowServerContext", _USE_MODULE_CONTEXT)
    if override is not _USE_MODULE_CONTEXT:
        return override  # type: ignore[return-value]
    return mlflow_adapter_module.MLflowServerContext


def _resolve_server_adapter_cls():
    override = globals().get("MLflowServerAdapter", _USE_MODULE_ADAPTER)
    if override is not _USE_MODULE_ADAPTER:
        return override  # type: ignore[return-value]
    return mlflow_adapter_module.MLflowServerAdapter


@dataclass
class MLflowResourceState:
    """State container for MLflow resources.

    Thread Safety:
        The active_run_stack is protected by a threading.Lock to ensure
        atomic operations in multi-threaded environments. All stack access
        must be synchronized through the lock.
    """

    client: MlflowClient | None = None
    server_context: Any = None  # Type is dynamically resolved
    server_info: Any = None
    active_run_stack: list[str] = field(default_factory=list)  # Stack of active run IDs for nesting
    stack_lock: threading.Lock = field(default_factory=threading.Lock)  # Protects active_run_stack
    experiment_id: str | None = None
    cleanup_callbacks: list[Any] = field(default_factory=list)  # Callbacks can be any callable
    server_start_error: Exception | None = None
    tracking_uri: str | None = None  # Track set tracking URI for conflict detection


class MLflowResourceManager:
    """Centralized resource manager for MLflow components.

    This class coordinates the lifecycle of all MLflow resources:
    - MLflow client instances
    - Server contexts and processes
    - Active runs and experiments
    - Cleanup operations

    Follows the Resource Manager Pattern to ensure guaranteed cleanup.

    Global State Dependencies:
        This manager coordinates both MlflowClient instances (explicit tracking URI)
        and global MLflow state for compatibility with Lightning and nested runs:

        - mlflow.set_tracking_uri(): Set during initialization to sync with client
        - mlflow.active_run(): Used for nested run detection
        - mlflow.end_run(): Used during cleanup
        - Lightning's MLFlowLogger: Depends on global tracking URI for artifacts

        The global tracking URI is automatically set when creating the client to
        ensure consistency between client-based operations and global API calls.
        This prevents artifact logging failures when using mlflow-artifacts:// URIs.
    """

    def __init__(self, mlflow_config: MLflowSettings | None = None):
        """Initialize resource manager.

        Args:
            mlflow_config: MLflow configuration settings
        """
        self._config = mlflow_config
        self._state = MLflowResourceState()
        self._is_initialized = False

    def __enter__(self) -> MLflowResourceManager:
        """Enter context and initialize resources."""
        if self._is_initialized:
            raise RuntimeError("Resource manager already initialized")

        self._initialize_resources()
        self._is_initialized = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and clean up all resources."""
        self._cleanup_all_resources()
        self._is_initialized = False

    def _initialize_resources(self) -> None:
        """Initialize MLflow resources based on configuration.

        Raises:
            RuntimeError: If conflicting tracking URIs are detected between
                server_info and client_config.
        """
        if not self._config or not getattr(self._config, "enabled", False):
            logger.debug("MLflow disabled, using minimal resource initialization")
            self._state.client = MLflowClientFactory.create_client()
            return

        # Initialize server if configured
        server_config = getattr(self._config, "server", None)
        client_config = getattr(self._config, "client", None)

        # Only start server if tracking URI is HTTP-based (not file://)
        should_start_server = False
        if server_config and client_config:
            tracking_uri = getattr(client_config, "tracking_uri", None)
            if tracking_uri:
                tracking_uri_str = str(tracking_uri)
                # Only start server for HTTP(S) URIs, not file:// URIs
                should_start_server = tracking_uri_str.startswith(("http://", "https://"))

        if should_start_server:
            self._initialize_server(server_config)

        # Detect and validate potential dual tracking URI sources
        server_uri = None
        client_uri = None

        if self._state.server_info and hasattr(self._state.server_info, "url"):
            server_uri = str(self._state.server_info.url)

        if client_config and hasattr(client_config, "tracking_uri") and client_config.tracking_uri:
            client_uri = str(client_config.tracking_uri)

        # Normalize URIs for comparison (strip trailing slashes)
        # This ensures 'http://host:port' and 'http://host:port/' are treated as equivalent
        normalized_server_uri = _normalize_tracking_uri(server_uri)
        normalized_client_uri = _normalize_tracking_uri(client_uri)

        # Conflict detection: both sources exist but URIs differ
        if normalized_server_uri and normalized_client_uri and normalized_server_uri != normalized_client_uri:
            msg = (
                f"Conflicting tracking URIs detected:\n"
                f"  Server URI: {server_uri}\n"
                f"  Client URI: {client_uri}\n"
                f"Both server_info and client_config specify tracking URIs but they differ. "
                f"Please ensure consistent configuration."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # Redundancy warning: both sources exist with same URI (after normalization)
        if normalized_server_uri and normalized_client_uri and normalized_server_uri == normalized_client_uri:
            logger.warning(
                f"Redundant tracking URI configuration detected: {server_uri}. "
                f"Both server_info and client_config specify the same URI. "
                f"Consider removing one source for clarity."
            )

        # Initialize client
        if self._state.server_info:
            logger.debug(f"Creating client from server info: {self._state.server_info}")
            self._state.client = MLflowClientFactory.create_client_from_server_info(
                self._state.server_info, client_config
            )
            tracking_uri = getattr(self._state.client, "tracking_uri", "unknown")
            logger.debug(f"Client created with tracking URI: {tracking_uri}")

            # Set global tracking URI for consistency with Lightning and artifact logging
            if server_uri:
                self._set_global_tracking_uri(server_uri)
        else:
            logger.debug(f"Creating client from config: {client_config}")
            self._state.client = MLflowClientFactory.create_client(client_config)
            tracking_uri = getattr(self._state.client, "tracking_uri", "unknown")
            logger.debug(f"Client created with tracking URI: {tracking_uri}")

            # Set global tracking URI for consistency with Lightning and artifact logging
            if client_uri:
                self._set_global_tracking_uri(client_uri)

        # Skip client validation if we just started our own server
        # (server may still be initializing, validation will happen on first use)
        if self._state.server_info:
            logger.debug("Skipping client validation for newly started server")
        else:
            # Only validate connectivity for external servers or file-based tracking
            if not MLflowClientFactory.validate_client_connectivity(self._state.client):
                logger.warning("MLflow client connectivity validation failed")

    def _set_global_tracking_uri(self, uri: str) -> None:
        """Set global MLflow tracking URI with conflict detection.

        Args:
            uri: Tracking URI to set

        Raises:
            RuntimeError: If attempting to set a different URI than already set
        """
        current_uri = self._state.tracking_uri
        if current_uri and current_uri != uri:
            msg = (
                f"Attempting to change global tracking URI from {current_uri} to {uri}. "
                f"This indicates a configuration conflict or improper state management."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        logger.debug(f"Setting global tracking URI: {uri}")
        mlflow.set_tracking_uri(uri)
        self._state.tracking_uri = uri

    def _validate_stack_consistency(self) -> None:
        """Validate internal run stack consistency with MLflow global state.

        This method checks if our internal active_run_stack is synchronized
        with MLflow's global mlflow.active_run() state. Any desynchronization
        is logged as a warning to aid debugging.

        Thread Safety:
            Must be called while holding stack_lock.
        """
        # Get MLflow's global active run
        global_active = mlflow.active_run()

        # Check stack consistency
        with self._state.stack_lock:
            if self._state.active_run_stack:
                internal_active = self._state.active_run_stack[-1]
                if global_active:
                    global_run_id = global_active.info.run_id
                    if internal_active != global_run_id:
                        logger.warning(
                            f"Stack desynchronization detected:\n"
                            f"  Internal stack top: {internal_active}\n"
                            f"  MLflow global run: {global_run_id}\n"
                            f"This may indicate improper run lifecycle management."
                        )
                else:
                    logger.warning(
                        f"Stack desynchronization detected:\n"
                        f"  Internal stack top: {internal_active}\n"
                        f"  MLflow global run: None\n"
                        f"Internal stack has active run but MLflow global state is empty."
                    )
            else:
                # Stack is empty - global state should also be empty
                if global_active:
                    logger.warning(
                        f"Stack desynchronization detected:\n"
                        f"  Internal stack: empty\n"
                        f"  MLflow global run: {global_active.info.run_id}\n"
                        f"MLflow has active run but internal stack is empty."
                    )

    def _get_state_snapshot(self) -> dict[str, Any]:
        """Get current state snapshot for debugging.

        Returns:
            Dictionary containing current state information
        """
        with self._state.stack_lock:
            global_run = mlflow.active_run()
            return {
                "initialized": self._is_initialized,
                "tracking_uri": self._state.tracking_uri,
                "client_exists": self._state.client is not None,
                "server_running": self._state.server_info is not None,
                "active_run_stack": list(self._state.active_run_stack),
                "stack_depth": len(self._state.active_run_stack),
                "experiment_id": self._state.experiment_id,
                "mlflow_global_run": global_run.info.run_id if global_run else None,
            }

    def _initialize_server(self, server_config: Any) -> None:
        """Initialize MLflow server - stores context for later cleanup."""
        try:
            health_checker = health_checker_module.CompositeHealthChecker(
                health_checker_module.HTTPHealthChecker(
                    health_endpoint="/",
                    request_timeout=1.0,
                    wait_timeout=15.0,
                    poll_interval=0.5,
                ),
                health_checker_module.MLflowAPIHealthChecker(
                    request_timeout=1.0,
                    wait_timeout=15.0,
                    poll_interval=0.5,
                ),
            )

            adapter_cls = _resolve_server_adapter_cls()
            configured_timeout = getattr(server_config, "health_timeout", None)
            timeout_seconds = float(configured_timeout) if configured_timeout is not None else 30.0
            timeout_seconds = max(timeout_seconds, 20.0)
            adapter = adapter_cls(
                health_checker=health_checker,
                # Honor settings-defined timeout while enforcing a sensible minimum
                health_timeout=timeout_seconds,
            )

            # Create and start server context
            # Note: We can't use 'with' here because the context must live
            # for the lifetime of the resource manager
            logger.debug("Starting MLflow server context")
            context_cls = _resolve_server_context_cls()
            server_context = context_cls(
                server_config=server_config,
                adapter=adapter,
            )

            # Start the server and store both context and info. Support
            # context managers that expose ``__enter__`` but not
            # ``start_server`` (used heavily in tests).
            if hasattr(server_context, "start_server"):
                server_info = server_context.start_server()
            elif hasattr(server_context, "__enter__"):
                server_info = server_context.__enter__()  # type: ignore[call-arg]
            else:
                msg = (
                    "MLflow server context must provide either 'start_server'"
                    " or context manager protocols"
                )
                raise RuntimeError(msg)

            self._state.server_context = server_context
            self._state.server_info = server_info
            self._state.server_start_error = None

            # Register cleanup callback
            self._state.cleanup_callbacks.append(self._cleanup_server)

            logger.info(f"MLflow server started successfully: {self._state.server_info.url}")

        except Exception as e:
            logger.error(f"Failed to initialize MLflow server: {e}")
            self._state.server_context = None
            self._state.server_info = None
            self._state.server_start_error = e
            raise RuntimeError("Failed to initialize MLflow server") from e

    def get_client(self) -> MlflowClient:
        """Get MLflow client instance.

        Returns:
            Configured MlflowClient instance

        Raises:
            RuntimeError: If resource manager not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized - use as context manager")

        if self._state.client is None:
            raise RuntimeError("MLflow client not available")

        return self._state.client

    def get_server_info(self) -> Any:
        """Get server information if server was started.

        Returns:
            Server info or None if no server started
        """
        return self._state.server_info

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ):
        """Create MLflow run with proper resource management.

        Thread Safety:
            All stack operations are protected by stack_lock. Multiple threads
            can safely create nested runs without race conditions.

        Args:
            experiment_name: Name of experiment
            run_name: Name of run
            nested: Whether to create nested run

        Yields:
            Run context for logging operations

        Raises:
            RuntimeError: If resource manager not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Resource manager not initialized")

        # Validate stack consistency before creating run
        self._validate_stack_consistency()

        client = self.get_client()
        run_id = None

        try:
            # Get or create experiment
            exp_name = experiment_name or "DLKit"
            logger.debug(f"Getting or creating experiment '{exp_name}'")
            experiment_id = MLflowClientFactory.get_or_create_experiment(client, exp_name)
            logger.debug(f"Experiment created/found with ID: {experiment_id}")
            self._state.experiment_id = experiment_id

            # Thread-safe stack access to determine parent run
            parent_run_id = None
            with self._state.stack_lock:
                if nested and self._state.active_run_stack:
                    # Use the top of the stack as the parent run
                    parent_run_id = self._state.active_run_stack[-1]
                    logger.debug(f"Creating nested run under parent: {parent_run_id}")

            # Create run using client with correct parameters
            # Note: MlflowClient.create_run uses different parameter names than mlflow.start_run
            run_kwargs: dict[str, Any] = {
                "experiment_id": experiment_id,
            }
            if run_name:
                # MLflow client expects 'tags' parameter with a special tag for run name
                run_kwargs["tags"] = {"mlflow.runName": run_name}
            if parent_run_id:
                # MLflow client expects 'tags' parameter with parent run tag
                if "tags" not in run_kwargs:
                    run_kwargs["tags"] = {}
                run_kwargs["tags"]["mlflow.parentRunId"] = parent_run_id

            run = client.create_run(**run_kwargs)
            run_id = run.info.run_id

            # Thread-safe stack push
            with self._state.stack_lock:
                self._state.active_run_stack.append(run_id)
                logger.debug(f"Created MLflow run: {run_id}, stack depth: {len(self._state.active_run_stack)}")

            # Yield run context for operations
            from .mlflow_run_context import ClientBasedRunContext
            yield ClientBasedRunContext(client, run_id)

        finally:
            # Clean up run
            if run_id:
                try:
                    client.set_terminated(run_id, status="FINISHED")
                    logger.debug(f"Terminated MLflow run: {run_id}")
                except Exception as e:
                    logger.warning(f"Failed to terminate run {run_id}: {e}")
                finally:
                    # Thread-safe stack pop
                    with self._state.stack_lock:
                        if self._state.active_run_stack and self._state.active_run_stack[-1] == run_id:
                            self._state.active_run_stack.pop()
                            logger.debug(f"Popped run {run_id} from stack, stack size: {len(self._state.active_run_stack)}")

                    # Clear active run from MLflow's global state
                    try:
                        mlflow.end_run()
                    except Exception:
                        pass

                    # Validate stack consistency after run cleanup
                    self._validate_stack_consistency()

    def add_cleanup_callback(self, callback: Any) -> None:
        """Add cleanup callback to be executed during resource cleanup.

        Args:
            callback: Function to call during cleanup
        """
        self._state.cleanup_callbacks.append(callback)

    def _cleanup_server(self) -> None:
        """Clean up server context."""
        context = self._state.server_context
        if context:
            try:
                logger.debug("Cleaning up MLflow server context")
                if hasattr(context, "stop_server"):
                    context.stop_server()  # type: ignore[call-arg]
                elif hasattr(context, "__exit__"):
                    context.__exit__(None, None, None)  # type: ignore[call-arg]
            except Exception as e:
                logger.warning(f"Error during server cleanup: {e}")
            finally:
                self._state.server_context = None
                self._state.server_info = None

    def _cleanup_all_resources(self) -> None:
        """Clean up all managed resources.

        Thread Safety:
            Uses stack_lock to ensure atomic cleanup of active runs.
        """
        logger.debug("Starting MLflow resource cleanup")

        # Clean up active runs if any (in reverse order) - thread-safe
        with self._state.stack_lock:
            if self._state.active_run_stack and self._state.client:
                runs_to_cleanup = list(reversed(self._state.active_run_stack))
                for run_id in runs_to_cleanup:
                    try:
                        self._state.client.set_terminated(run_id, status="FINISHED")
                        logger.debug(f"Terminated active run: {run_id}")
                    except Exception as e:
                        logger.warning(f"Failed to terminate active run {run_id}: {e}")
                self._state.active_run_stack.clear()

        # Clear any lingering global MLflow state (safety cleanup)
        # Note: We already call mlflow.end_run() in create_run's finally block,
        # but this ensures cleanup even if external code started runs
        try:
            mlflow.end_run()  # Clear active run if any
        except Exception:
            pass  # No active run to clear

        # Execute cleanup callbacks in reverse order
        for callback in reversed(self._state.cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cleanup callback failed: {e}")

        # Reset MLflow global state to prevent leakage between tests
        self.reset_global_state()

        # Reset state
        self._state = MLflowResourceState()
        logger.debug("MLflow resource cleanup completed")

    @staticmethod
    def reset_global_state() -> None:
        """Reset MLflow global state for test isolation.

        This utility method helps with test isolation by clearing
        global MLflow state that may persist between test runs.
        """
        try:
            # Clear tracking URI - mlflow accepts None but type hints don't reflect it
            mlflow.set_tracking_uri(None)  # type: ignore[arg-type]

            # End any active run
            try:
                mlflow.end_run()
            except Exception:
                pass

            # Clear any cached state (internal MLflow attribute, may not exist)
            if hasattr(mlflow, "_active_run_stack"):
                mlflow._active_run_stack.clear()  # type: ignore[attr-defined]

            logger.debug("MLflow global state reset completed")

        except Exception as e:
            logger.warning(f"Failed to reset MLflow global state: {e}")
