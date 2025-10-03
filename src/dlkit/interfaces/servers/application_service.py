"""Application service layer orchestrating server management business logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .factory import create_mlflow_adapter
from .server_management_service import ServerManagementService
from .protocols import ServerInfo, ServerStatus, ServerAdapter

if TYPE_CHECKING:  # pragma: no cover
    from dlkit.tools.config.mlflow_settings import MLflowServerSettings
else:  # pragma: no cover - runtime fallback for typing only
    MLflowServerSettings = Any


class ServerApplicationService:
    """Application service for server management business logic (SRP: Orchestrates workflows)."""

    def __init__(
        self,
        server_adapter: ServerAdapter | None = None,
        server_management: ServerManagementService | None = None,
        server_config: Any = None,
    ) -> None:
        """Initialize application service.

        Args:
            server_adapter: Server adapter implementation
            server_management: Server management service
            server_config: Server configuration for timeout settings
        """
        if server_adapter is None:
            # Extract timeout settings from server config if provided
            kwargs = {}
            if server_config:
                if hasattr(server_config, "health_timeout"):
                    kwargs["health_timeout"] = server_config.health_timeout
                if hasattr(server_config, "request_timeout"):
                    kwargs["request_timeout"] = server_config.request_timeout
                if hasattr(server_config, "poll_interval"):
                    kwargs["poll_interval"] = server_config.poll_interval
            self._server_adapter = create_mlflow_adapter(**kwargs)
        else:
            self._server_adapter = server_adapter
        self._server_management = server_management or ServerManagementService()

    def start_server(
        self,
        config_path: Path | None = None,
        host: str | None = None,
        port: int | None = None,
        backend_store_uri: str | None = None,
        artifacts_destination: str | None = None,
    ) -> ServerInfo:
        """Start server with configuration and overrides.

        Args:
            config_path: Optional path to configuration file
            host: Override server hostname
            port: Override server port
            backend_store_uri: Override backend store URI
            artifacts_destination: Override artifacts destination

        Returns:
            ServerInfo containing server details

        Raises:
            RuntimeError: If server cannot be started
        """
        # Load configuration
        server_config = self._load_server_configuration(
            config_path, host, port, backend_store_uri, artifacts_destination
        )

        # Setup storage (CLI layer handles user interaction)
        overrides = self._build_overrides_dict(host, port, backend_store_uri, artifacts_destination)
        final_config = self._server_management.ensure_storage_setup(server_config, overrides)

        # Start server
        server_info = self._server_adapter.start_server(
            server_config=final_config, **{k: v for k, v in overrides.items() if v is not None}
        )

        # Track the server
        if server_info.pid:
            self._server_management.track_server(
                server_info.host, server_info.port, server_info.pid
            )

        return server_info

    def stop_server(
        self, host: str = "localhost", port: int = 5000, force: bool = False
    ) -> tuple[bool, list[str]]:
        """Stop server at given host:port.

        Args:
            host: Server hostname
            port: Server port
            force: Force stop even if server is not responding

        Returns:
            Tuple of (success, status_messages)
        """
        # Check if server is running first (unless force)
        if not force:
            try:
                status = self._server_adapter.check_server(host, port)
                if not status.is_running:
                    return True, [f"No server found running at {host}:{port}"]
            except Exception as e:
                return False, [f"Could not check server status: {e}"]

        # Stop processes
        success, messages = self._server_management.stop_server_processes(host, port, force)

        # Remove from tracking if successful
        if success:
            self._server_management.untrack_server(host, port)

        return success, messages

    def check_server_status(self, host: str = "localhost", port: int = 5000) -> ServerStatus:
        """Check server status.

        Args:
            host: Server hostname
            port: Server port

        Returns:
            ServerStatus containing server information
        """
        return self._server_adapter.check_server(host, port)

    def get_server_configuration_info(self, config_path: Path | None = None) -> dict[str, Any]:
        """Get server configuration information.

        Args:
            config_path: Optional path to configuration file

        Returns:
            Dictionary containing configuration details
        """
        if not config_path:
            return {"configured": False, "message": "No configuration file provided"}

        try:
            from dlkit.interfaces.cli.adapters.config_adapter import load_config

            settings = load_config(config_path)

            if not settings.MLFLOW or not settings.MLFLOW.enabled:
                return {
                    "configured": False,
                    "message": "MLflow not configured or not enabled",
                }

            mlflow_config = settings.MLFLOW
            return {
                "configured": True,
                "server": {
                    "host": mlflow_config.server.host,
                    "port": mlflow_config.server.port,
                    "backend_store": str(mlflow_config.server.backend_store_uri),
                    "artifacts": str(mlflow_config.server.artifacts_destination),
                },
                "client": {
                    "tracking_uri": str(mlflow_config.client.tracking_uri),
                    "experiment": mlflow_config.client.experiment_name,
                },
            }

        except Exception as e:
            return {
                "configured": False,
                "message": f"Error loading configuration: {e}",
            }

    def _load_server_configuration(
        self,
        config_path: Path | None,
        host: str | None,
        port: int | None,
        backend_store_uri: str | None,
        artifacts_destination: str | None,
    ) -> MLflowServerSettings:
        """Load server configuration from file or create defaults.

        Args:
            config_path: Optional configuration file path
            host: Host override
            port: Port override
            backend_store_uri: Backend store URI override
            artifacts_destination: Artifacts destination override

        Returns:
            MLflowServerSettings object (always consistent type)
        """
        if config_path is not None:
            from dlkit.interfaces.cli.adapters.config_adapter import load_config

            settings = load_config(config_path)
            if settings.MLFLOW is not None:
                # Return the server settings directly, not a context wrapper
                return settings.MLFLOW.server

        # Create default configuration
        from dlkit.tools.config.mlflow_settings import MLflowServerSettings

        return MLflowServerSettings(
            host=(host or "127.0.0.1"),
            port=(port or 5000),
            backend_store_uri=backend_store_uri,
            artifacts_destination=artifacts_destination,
        )

    def _build_overrides_dict(
        self,
        host: str | None,
        port: int | None,
        backend_store_uri: str | None,
        artifacts_destination: str | None,
    ) -> dict[str, Any]:
        """Build overrides dictionary from parameters.

        Args:
            host: Host override
            port: Port override
            backend_store_uri: Backend store URI override
            artifacts_destination: Artifacts destination override

        Returns:
            Dictionary of non-None overrides
        """
        overrides = {}
        if host is not None:
            overrides["host"] = host
        if port is not None:
            overrides["port"] = port
        if backend_store_uri is not None:
            overrides["backend_store_uri"] = backend_store_uri
        if artifacts_destination is not None:
            overrides["artifacts_destination"] = artifacts_destination
        return overrides
