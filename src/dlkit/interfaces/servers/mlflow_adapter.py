"""MLflow server adapter implementation."""

from __future__ import annotations

from typing import Any
import os
from subprocess import Popen
from pathlib import Path

from urllib.parse import urlparse

from dlkit.tools.utils.logging_config import get_logger
from dlkit.tools.io import locations
from dlkit.tools.config.mlflow_settings import MLflowServerSettings
from dlkit.tools.utils.system_utils import mkdir_for_local

from .health_checker import HTTPHealthChecker
from .process_manager import SubprocessManager
from .protocols import (
    ContextualServerAdapter,
    HealthChecker,
    ProcessManager,
    ServerInfo,
    ServerStatus,
)

logger = get_logger(__name__)


class _ServerProcessTracker:
    """Internal process tracking for server lifecycle management.

    This class is not part of the public API and handles the separation
    between server information (dataflow) and process management (behavior).
    """

    def __init__(self, process_manager: ProcessManager) -> None:
        """Initialize process tracker.

        Args:
            process_manager: Process manager for lifecycle operations
        """
        self._processes: dict[str, Popen[bytes]] = {}
        self._process_manager = process_manager

    def register(self, server_info: ServerInfo, process: Popen[bytes]) -> str:
        """Register a server process for tracking.

        Args:
            server_info: Server information
            process: Process handle

        Returns:
            Handle ID for future reference
        """
        handle_id = f"{server_info.host}:{server_info.port}"
        self._processes[handle_id] = process
        logger.debug(f"Registered server process: {handle_id} -> PID {process.pid}")
        return handle_id

    def cleanup(self, handle_id: str) -> bool:
        """Clean up a registered server process.

        Args:
            handle_id: Handle ID from register()

        Returns:
            True if cleanup successful
        """
        if handle_id in self._processes:
            process = self._processes.pop(handle_id)
            logger.debug(f"Cleaning up server process: {handle_id} -> PID {process.pid}")
            return self._process_manager.stop_process(process)
        logger.debug(f"No process found for handle: {handle_id}")
        return True

    def has_process(self, handle_id: str) -> bool:
        """Check if a process is registered.

        Args:
            handle_id: Handle ID to check

        Returns:
            True if process is registered
        """
        return handle_id in self._processes


class MLflowServerAdapter(ContextualServerAdapter):
    """MLflow server adapter with proper separation of concerns."""

    def __init__(
        self,
        process_manager: ProcessManager | None = None,
        health_checker: HealthChecker | None = None,
        scheme: str = "http",
        health_timeout: float | None = None,
        request_timeout: float | None = None,
        poll_interval: float | None = None,
    ) -> None:
        """Initialize MLflow server adapter.

        Args:
            process_manager: Process manager for server lifecycle
            health_checker: Health checker for server status
            scheme: URL scheme (http or https)
            health_timeout: Timeout for health checks in seconds (defaults to MLflowServerSettings default)
            request_timeout: Timeout for individual health check requests in seconds (defaults to MLflowServerSettings default)
            poll_interval: Interval between health check polls in seconds (defaults to MLflowServerSettings default)
        """
        # Use MLflowServerSettings as single source of truth for defaults
        _defaults = MLflowServerSettings()
        self._health_timeout = float(health_timeout or _defaults.health_timeout)
        self._request_timeout = float(request_timeout or _defaults.request_timeout)
        self._poll_interval = float(poll_interval or _defaults.poll_interval)

        # Always keep a persistent process manager so process lifetimes are tied to
        # the adapter rather than temporary helper instances. This prevents early
        # garbage collection (and the resulting SIGTERM) for servers we launch
        # ourselves while still honouring dependency injection when callers supply
        # a custom manager.
        self._process_manager = process_manager or SubprocessManager()
        if health_checker is not None:
            self._health_checker = health_checker
            self._has_custom_health_checker = True
        else:
            self._health_checker = self._create_default_health_checker()
            self._has_custom_health_checker = False
        self._scheme = scheme
        self._current_server_info: ServerInfo | None = None
        # NEW: Internal process tracking for SOLID compliance
        self._process_tracker = _ServerProcessTracker(self._process_manager)

    def start_server(
        self,
        server_config: MLflowServerSettings,
        **overrides: Any,
    ) -> ServerInfo:
        """Start MLflow server with optional overrides.

        Args:
            server_config: MLflow server configuration
            **overrides: Server configuration overrides (host, port, etc.)

        Returns:
            ServerInfo containing server information
        """
        try:
            # Apply server overrides if provided
            if overrides:
                logger.debug("Applying server configuration overrides", overrides=overrides)
                server_config = self._apply_server_overrides(server_config, overrides)

            server_config = self._normalize_server_config(server_config)

            # Pre-check: if an instance is already running, reuse it immediately
            server_url = self.get_server_url(server_config.host, server_config.port)
            logger.info(f"Starting MLflow server at {server_url}")
            logger.debug("Checking if server already running", url=server_url)
            # Use a fresh default health checker unless a custom one was provided
            if not self._has_custom_health_checker:
                self._health_checker = self._create_default_health_checker()
            health_checker = self._health_checker
            status = health_checker.check_health(server_url)
            if status.is_running:
                server_info = ServerInfo(
                    process=None,
                    url=server_url,
                    host=server_config.host,
                    port=server_config.port,
                )
                logger.info(
                    "MLflow server already running - reusing existing instance",
                    url=server_url,
                    host=server_config.host,
                    port=server_config.port,
                )
                return server_info

            # Default storage if not provided and ensure local dirs
            logger.debug("Resolving storage locations and ensuring directories")
            try:
                if not server_config.backend_store_uri:
                    server_config = server_config.model_copy(
                        update={"backend_store_uri": locations.mlruns_backend_uri()}
                    )
                if not server_config.artifacts_destination:
                    server_config = server_config.model_copy(
                        update={"artifacts_destination": f"file://{locations.mlartifacts_dir()}"}
                    )
                # Config is already normalized at line 154, no need to normalize again
                self._ensure_local_storage(server_config)
            except Exception as e:
                # Best-effort; fall back to MLflow defaults
                logger.debug(f"Failed to ensure local storage: {e}", exc_info=True)
                pass

            # Start new server process
            logger.debug(
                "Creating MLflow server process",
                host=server_config.host,
                port=server_config.port,
                backend_store=str(server_config.backend_store_uri),
                artifacts_dest=str(server_config.artifacts_destination),
            )
            process = self._process_manager.start_process(server_config)

            # Wait for server to become healthy
            logger.debug("Waiting for MLflow server health check", timeout=self._health_timeout)
            if not health_checker.wait_for_health(server_url, timeout=self._health_timeout):
                logger.warning(
                    "MLflow server failed health check - stopping process and checking existing instance"
                )
                try:
                    self._process_manager.stop_process(process)
                except Exception:
                    pass
                # If another instance is already running, reuse it
                status = health_checker.check_health(server_url)
                if status.is_running:
                    server_info = ServerInfo(
                        process=None,
                        url=server_url,
                        host=server_config.host,
                        port=server_config.port,
                    )
                    logger.info(
                        "MLflow server already running - reusing existing instance",
                        url=server_url,
                        host=server_config.host,
                        port=server_config.port,
                    )
                    return server_info
                raise RuntimeError(
                    f"MLflow server failed health check within {self._health_timeout}s at {server_url}"
                )

            # Create ServerInfo with process separation for SOLID compliance
            server_info = ServerInfo(
                process=process,
                url=server_url,
                host=server_config.host,
                port=server_config.port,
                pid=process.pid if hasattr(process, "pid") else None,
            )

            # NEW: Internal process tracking
            handle_id = self._process_tracker.register(server_info, process)
            server_info._internal_handle_id = handle_id  # Internal use only

            logger.info(
                "MLflow server started successfully",
                url=server_url,
                pid=server_info.pid,
                backend_store=str(server_config.backend_store_uri),
                artifacts_dest=str(server_config.artifacts_destination),
            )

            return server_info

        except Exception as e:
            logger.error("Failed to start MLflow server", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to start MLflow server: {str(e)}")

    def stop_server(self, server_info: ServerInfo) -> bool:
        """Stop MLflow server.

        Args:
            server_info: Information about the server to stop

        Returns:
            True if server was stopped successfully
        """
        logger.debug("Starting MLflow server shutdown")
        try:
            # Use internal process tracking first (SOLID compliance)
            handle_id = getattr(server_info, "_internal_handle_id", None)
            if handle_id and self._process_tracker.has_process(handle_id):
                logger.debug(f"Stopping server using internal tracking: {handle_id}")
                success = self._process_tracker.cleanup(handle_id)
                if success:
                    logger.info("MLflow server stopped", url=server_info.url)
                    return True

            # Fallback to ServerInfo.process
            if server_info.process is not None:
                logger.debug("Falling back to process handle")
                success = self._process_manager.stop_process(server_info.process)
                if success:
                    logger.info("MLflow server stopped", url=server_info.url)
                    return True
                else:
                    # Process exists but stopping failed
                    raise RuntimeError("Failed to stop MLflow server process")

            # No process handle available
            logger.warning(
                "Cannot stop MLflow server - no process handle available",
                url=server_info.url,
                pid=server_info.pid,
            )
            raise RuntimeError("Cannot stop server: no process handle available")

        except Exception as e:
            logger.error(
                "Error stopping MLflow server",
                error=str(e),
                pid=server_info.pid,
                url=server_info.url,
                exc_info=True,
            )
            raise RuntimeError(f"Failed to stop MLflow server: {str(e)}")

    def check_server(
        self,
        host: str = "localhost",
        port: int = 5000,
    ) -> ServerStatus:
        """Check MLflow server status.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            ServerStatus containing server status
        """
        try:
            url = self.get_server_url(host, port)
            if not self._has_custom_health_checker:
                self._health_checker = self._create_default_health_checker()
            health_checker = self._health_checker
            status = health_checker.check_health(url)
            return status

        except Exception as e:
            raise RuntimeError(f"Failed to check server status: {str(e)}")

    def get_server_url(self, host: str, port: int) -> str:
        """Get MLflow server URL.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            Complete server URL
        """
        return f"{self._scheme}://{host}:{port}"

    def __enter__(self) -> ServerInfo:
        """Enter context and start server.

        Returns:
            ServerInfo: Information about the started server

        Raises:
            RuntimeError: If server start fails
        """
        if self._current_server_info is not None:
            raise RuntimeError("Server already started in this context")
        # No default settings available here
        raise NotImplementedError(
            "Context manager requires settings. Use MLflowServerContext instead."
        )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and stop server."""
        if self._current_server_info is not None:
            try:
                _ = self.stop_server(self._current_server_info)
            except Exception as e:
                logger.error(f"Failed to stop server in context: {e}")
            self._current_server_info = None

    def _create_default_health_checker(self) -> HealthChecker:
        """Create a fresh health checker using current timeout configuration."""
        return HTTPHealthChecker(
            request_timeout=self._request_timeout,
            wait_timeout=self._health_timeout,
            poll_interval=self._poll_interval,
        )

    @staticmethod
    def _apply_server_overrides(
        server_config: MLflowServerSettings, overrides: dict[str, Any]
    ) -> MLflowServerSettings:
        """Apply server configuration overrides.

        Args:
            server_config: Base server configuration
            overrides: Override values

        Returns:
            Updated server configuration
        """
        override_dict = {}
        for key in ["host", "port", "backend_store_uri", "artifacts_destination"]:
            if key in overrides:
                override_dict[key] = overrides[key]

        return server_config.model_copy(update=override_dict)

    def _ensure_local_storage(self, config: MLflowServerSettings) -> None:
        """Create local storage directories for backend & artifacts if needed.

        Args:
            config: MLflow server configuration

        Raises:
            RuntimeError: If directory creation fails
        """
        logger.debug("Ensuring local storage directories")
        local_hosts = {"localhost", "127.0.0.1", "::1", None}

        for attr in ("backend_store_uri", "artifacts_destination"):
            uri = getattr(config, attr)
            logger.debug(f"Checking {attr} = {uri}")
            if uri is None:
                continue

            try:
                parsed = urlparse(str(uri))
            except Exception:
                parsed = None

            is_file = bool(parsed and (parsed.scheme in ("", "file")))
            host_local = bool(parsed and (parsed.hostname in local_hosts))
            logger.debug(f"{attr} is_file={is_file}, host_local={host_local}")

            if is_file or host_local:
                try:
                    # mkdir_for_local expects an AnyUrl-like object or a file:// URL string
                    to_pass = uri
                    if not hasattr(uri, "path"):
                        uri_str = str(uri)
                        # Only add file:// prefix if it doesn't already have a scheme
                        if "://" not in uri_str:
                            to_pass = f"file://{_ensure_posix_path(uri_str)}"
                        else:
                            # For sqlite:// URIs, extract the path component
                            if uri_str.startswith("sqlite:///"):
                                # Extract path from sqlite:///path
                                path = uri_str.replace("sqlite:///", "")
                                to_pass = f"file:///{path}"
                            else:
                                to_pass = uri_str
                    logger.debug(f"Creating directory for {attr}: {to_pass}")
                    mkdir_for_local(to_pass)  # type: ignore[arg-type]
                    logger.debug(f"Successfully created directory for {attr}")
                except Exception as e:
                    logger.error(f"Failed to create local directory for {attr}: {e}")
                    raise RuntimeError(f"Directory creation failed for {attr}: {e}") from e

        logger.debug("Local storage directories ensured successfully")

    def _normalize_server_config(self, config: MLflowServerSettings) -> MLflowServerSettings:
        """Normalize server paths relative to current root context."""

        try:
            root_dir = locations.root()
        except Exception:
            root_dir = Path.cwd()

        updates: dict[str, Any] = {}

        backend_uri = getattr(config, "backend_store_uri", None)
        if backend_uri:
            backend_str = str(backend_uri)
            normalized_backend = self._normalize_backend_uri(backend_str, root_dir)
            if normalized_backend != backend_str:
                updates["backend_store_uri"] = normalized_backend

        artifacts_dest = getattr(config, "artifacts_destination", None)
        if artifacts_dest:
            artifacts_str = str(artifacts_dest)
            normalized_artifacts = self._normalize_artifacts_destination(artifacts_str, root_dir)
            if normalized_artifacts != artifacts_str:
                updates["artifacts_destination"] = normalized_artifacts

        if updates:
            return config.model_copy(update=updates)
        return config

    @staticmethod
    def _normalize_backend_uri(uri: str, root_dir: Path) -> str:
        parsed = urlparse(uri)
        if parsed.scheme not in {"file", "sqlite"}:
            return uri

        path = Path(parsed.path)
        if path.is_absolute():
            return uri

        resolved = (root_dir / path).resolve()
        if parsed.scheme == "sqlite":
            return f"sqlite:///{resolved.as_posix()}"
        return resolved.as_uri()

    @staticmethod
    def _normalize_artifacts_destination(destination: str, root_dir: Path) -> str:
        parsed = urlparse(destination)

        if parsed.scheme in {"", None}:
            path = Path(destination)
            if path.is_absolute():
                return destination
            return str((root_dir / path).resolve())

        if parsed.scheme == "file":
            path = Path(parsed.path)
            if path.is_absolute():
                return destination
            resolved = (root_dir / path).resolve()
            return resolved.as_uri()

        return destination


def _ensure_posix_path(p: str) -> str:
    try:
        return os.fspath(os.path.normpath(p)).replace("\\", "/")
    except Exception:
        return p


class MLflowServerContext:
    """Context manager wrapper for MLflow server adapter with server config."""

    def __init__(
        self,
        server_config: MLflowServerSettings,
        adapter: MLflowServerAdapter | None = None,
        **overrides: Any,
    ) -> None:
        """Initialize context with server config and overrides.

        Args:
            server_config: MLflow server configuration
            adapter: Server adapter to use
            **overrides: Server configuration overrides
        """
        self._server_config = server_config
        self._adapter = adapter or MLflowServerAdapter()
        self._overrides = overrides
        self._server_info: ServerInfo | None = None

    def __enter__(self) -> ServerInfo:
        """Enter context and start server.

        Returns:
            ServerInfo: Information about the started server

        Raises:
            RuntimeError: If server start fails
        """
        server_info = self._adapter.start_server(self._server_config, **self._overrides)
        self._server_info = server_info
        return server_info

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and stop server."""
        self.stop_server()

    def start_server(self) -> ServerInfo:
        """Start the MLflow server.

        Returns:
            ServerInfo: Information about the started server

        Raises:
            RuntimeError: If server start fails
        """
        server_info = self._adapter.start_server(self._server_config, **self._overrides)
        self._server_info = server_info
        return server_info

    def stop_server(self) -> None:
        """Stop the MLflow server."""
        if self._server_info is not None:
            try:
                success = self._adapter.stop_server(self._server_info)
                if not success:
                    logger.warning("Server stop returned False, but continuing cleanup")
            except Exception as e:
                logger.error(f"Failed to stop server: {e}")
            finally:
                self._server_info = None
