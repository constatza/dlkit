"""MLflow settings - flattened top-level configuration."""

from __future__ import annotations

import os
from pydantic import Field, field_validator, model_validator

from .core.base_settings import BasicSettings
from dlkit.tools.utils.system_utils import recommended_uvicorn_workers
from dlkit.core.datatypes.secure_uris import (
    SecureMLflowBackendStoreUri,
    SecureArtifactStoreUri,
    SecureMLflowTrackingUri,
)


class MLflowServerSettings(BasicSettings):
    """MLflow server configuration settings.

    Args:
        scheme: Server scheme (http/https)
        host: Server host address
        port: Server port number
        backend_store_uri: Backend store URI
        artifacts_destination: Artifacts storage path
        num_workers: Number of worker processes
        keep_alive_interval: Keep alive duration
        shutdown_timeout: Shutdown timeout
    """

    scheme: str = Field(default="http", description="MLflow server scheme (http/https only)")
    host: str = Field(default="127.0.0.1", description="MLflow server host address")
    port: int = Field(default=5000, description="MLflow server port number", gt=0, lt=65536)
    backend_store_uri: SecureMLflowBackendStoreUri | None = Field(
        default=None, description="URI for the backend store"
    )
    artifacts_destination: SecureArtifactStoreUri | None = Field(
        default=None, description="Default artifact root path (file://, s3://, or plain path)"
    )
    num_workers: int = Field(
        default=1,
        description="Number of worker processes. Note: SQLite backends require num_workers=1 "
        "due to concurrent write limitations. Use PostgreSQL/MySQL for multi-worker support.",
    )
    keep_alive_interval: int = Field(
        default=5, description="Duration in seconds to keep the server alive, after inactivity"
    )
    shutdown_timeout: int = Field(default=10, description="Timeout in seconds for server shutdown")
    health_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for health checks (accounts for DB migrations ~5s + startup ~2s + buffer)",
    )
    request_timeout: float = Field(
        default=2.0, description="Timeout for individual health check requests"
    )
    poll_interval: float = Field(default=0.5, description="Interval between health check polls")

    # Validators are provided by Secure* types; no additional validation required

    @field_validator("artifacts_destination", mode="before")
    @classmethod
    def _coerce_artifacts_destination(cls, v):
        # Leave plain paths as provided; coercion happens at adapter level
        return v

    @field_validator("scheme", mode="after")
    @classmethod
    def _validate_scheme(cls, value: str) -> str:
        """Ensure MLflow server scheme is HTTP-based.

        MLflow's built-in server accepts HTTP(S) URIs. Reject file:// and other
        schemes so configuration cannot request unsupported transports.
        """

        scheme = value.lower()
        if scheme not in {"http", "https"}:
            raise ValueError("MLflow server scheme must be 'http' or 'https'")
        return scheme

    @property
    def command(self) -> list[str]:
        """Generate MLflow server command.

        Returns:
            list[str]: Command line arguments for starting MLflow server
        """
        # Prefer explicitly configured workers when available; fall back to recommended formula
        workers = self.num_workers if self.num_workers > 0 else recommended_uvicorn_workers()
        command = [
            "mlflow",
            "server",
            "--host",
            str(self.host),
            "--port",
            str(self.port),
        ]
        # Only add URIs if provided
        if self.backend_store_uri is not None:
            command.extend(["--backend-store-uri", str(self.backend_store_uri)])
        if self.artifacts_destination is not None:
            command.extend(["--artifacts-destination", str(self.artifacts_destination)])
        if os.name != "nt":
            uvicorn_opts = [f"--workers {workers}"]
            if self.keep_alive_interval > 0:
                uvicorn_opts.append(f"--timeout-keep-alive {self.keep_alive_interval}")
            if self.shutdown_timeout > 0:
                uvicorn_opts.append(
                    f"--timeout-graceful-shutdown {self.shutdown_timeout}"
                )

            command.extend([
                "--uvicorn-opts",
                " ".join(uvicorn_opts),
            ])
        else:
            command.extend([
                "--waitress-opts",
                f"--threads={workers}",
            ])
        return command


class MLflowClientSettings(BasicSettings):
    """MLflow client configuration settings.

    Args:
        experiment_name: Name of MLflow experiment
        run_name: Optional name for MLflow run
        tracking_uri: MLflow tracking server URI
        register_model: Whether to register models
        max_trials: Max connection attempts
    """

    experiment_name: str = Field(default="Experiment", description="MLflow experiment name")
    run_name: str | None = Field(default=None, description="Name of the MLflow run")
    registered_model_name: str | None = Field(
        default=None,
        description="Optional override for registered model name",
    )
    registered_model_aliases: tuple[str, ...] | None = Field(
        default=None,
        description="Optional aliases to attach after registration",
    )
    registered_model_version_tags: dict[str, str] | None = Field(
        default=None,
        description="Optional tags to attach to each registered model version",
    )
    tracking_uri: SecureMLflowTrackingUri | None = Field(
        default=None, description="Tracking URI for MLflow"
    )
    register_model: bool = Field(default=True, description="Whether to register the model")
    max_trials: int = Field(
        default=3, description="Maximum number of trials for reaching mlflow server"
    )


class MLflowSettings(BasicSettings):
    """Top-level MLflow configuration for experiment tracking and model registration.

    Flattened from plugin architecture to top-level for easier access.
    Replaces: settings.SESSION.training.plugins["mlflow"]
    New usage: settings.MLFLOW

    Args:
        enabled: Whether MLflow tracking is enabled
        server: MLflow server configuration
        client: MLflow client configuration
    """

    enabled: bool = Field(default=False, description="Whether to enable MLflow tracking")
    server: MLflowServerSettings = Field(
        default_factory=MLflowServerSettings, description="MLflow server settings"
    )
    client: MLflowClientSettings = Field(
        default_factory=MLflowClientSettings, description="MLflow client settings"
    )

    @model_validator(mode="after")
    def validate_mlflow_config(self):
        """Validate MLflow configuration when enabled.

        The tracking URI may be None at config time — it is resolved at runtime to
        the SQLite filesystem default (``locations.mlruns_backend_uri()``) when not
        explicitly set.  This deferred resolution allows path-context-aware defaults
        without requiring a URI in every config file.
        """
        return self

    @field_validator("server", mode="after")
    @classmethod
    def _normalize_server_fields(cls, value: MLflowServerSettings) -> MLflowServerSettings:
        """Normalize server fields for test-friendly inputs.

        - Leave plain paths as-is for equality checks; CLI/adapter will coerce when starting.
        """
        try:
            return value
        except Exception:
            return value

    # (No additional validators at the MLflowSettings level)

    @property
    def experiment_name(self) -> str:
        """Get the experiment name for MLflow tracking.

        Returns:
            str: The experiment name
        """
        return self.client.experiment_name

    @property
    def run_name(self) -> str | None:
        """Get the run name for MLflow tracking.

        Returns:
            str | None: The run name if specified
        """
        return self.client.run_name

    @property
    def tracking_uri(self) -> str | None:
        """Get the MLflow tracking URI, or None if not explicitly configured.

        When None, the runtime resolver will default to the local SQLite store
        (``locations.mlruns_backend_uri()``).

        Returns:
            str | None: The tracking URI, or None for deferred resolution.
        """
        uri = self.client.tracking_uri
        return str(uri) if uri is not None else None
