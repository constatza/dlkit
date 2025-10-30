"""MLflow adapter implementing tracking abstractions."""

from contextlib import contextmanager, ExitStack, AbstractContextManager
from pathlib import Path
from typing import Any

import mlflow

from dlkit.interfaces.api.overrides.path_context import (
    get_current_path_context,
    path_override_context,
)
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.logging_config import get_logger

from .interfaces import IExperimentTracker, IRunContext
from .mlflow_resource_manager import MLflowResourceManager
from .mlflow_run_context import ClientBasedRunContext

logger = get_logger(__name__)

# MLflow integration constants
MLFLOW_DEFAULT_EXPERIMENT = "DLKit"
MLFLOW_TAG_SERVER_URL = "mlflow_server_url"
MLFLOW_TAG_SERVER_RUNNING = "mlflow_server_running"
MLFLOW_TAG_SERVER_LATENCY = "mlflow_server_response_time"
HEALTHCHECK_TIMEOUT_S = 0.2


class MLflowTracker(IExperimentTracker):
    """MLflow implementation of experiment tracker using resource manager pattern.

    Provides MLflow-based experiment tracking with proper resource lifecycle management
    through MLflowResourceManager. Handles MLflow server startup, client creation,
    experiment/run management, and guaranteed cleanup.

    The tracker should be used as a context manager to ensure proper resource cleanup:

    Example:
        ```python
        from dlkit.runtime.workflows.strategies.tracking import MLflowTracker

        tracker = MLflowTracker()
        tracker.setup_mlflow_config(settings.MLFLOW)

        with tracker:  # Initializes resources
            with tracker.create_run(experiment_name="training") as run:
                run.log_params({"learning_rate": 0.001})
                run.log_metrics({"loss": 0.5}, step=1)
                tracker.log_settings(settings, run)
        # Resources cleaned up automatically
        ```

    Attributes:
        disable_autostart (bool): Skip automatic MLflow server startup.
        skip_health_checks (bool): Skip server health validation checks.
    """

    def __init__(self, disable_autostart: bool = False, skip_health_checks: bool = False):
        """Initialize MLflow tracker.

        Args:
            disable_autostart: If True, skip automatic server startup. Useful when
                server is already running externally.
            skip_health_checks: If True, skip health validation checks. Useful for
                faster startup when server health is guaranteed.
        """
        self.disable_autostart = disable_autostart
        self.skip_health_checks = skip_health_checks
        self._resource_manager: MLflowResourceManager | None = None
        self._mlflow_config: Any = None
        self._exit_stack: ExitStack | None = None
        self._configured_server_url: str | None = None
        self._server_url: str | None = None
        self._server_status: dict | None = None
        self._root_dir: Path | None = None

    def __enter__(self) -> "MLflowTracker":
        """Context manager entry - initializes MLflow resources using ExitStack.

        Creates and enters the MLflowResourceManager which handles server startup,
        client creation, and experiment setup. Uses ExitStack for nested context
        management to ensure proper cleanup.

        Returns:
            MLflowTracker: Self for context manager protocol.

        Raises:
            Exception: If resource initialization fails. ExitStack ensures cleanup
                of any partially initialized resources.

        Example:
            ```python
            tracker = MLflowTracker()
            tracker.setup_mlflow_config(settings.MLFLOW)

            with tracker:  # Resources initialized here
                # Use tracker
                pass
            # Resources cleaned up here
            ```
        """
        logger.debug(f"MLflowTracker.__enter__ called - config={self._mlflow_config is not None}")

        if self._mlflow_config:
            try:
                # Create ExitStack for managing nested context managers
                self._exit_stack = ExitStack()
                self._exit_stack.__enter__()

                # Ensure MLflow resources resolve paths relative to root_dir overrides
                if self._root_dir is not None:
                    ctx = get_current_path_context()
                    has_root_override = bool(ctx and getattr(ctx, "root_dir", None))
                    if not has_root_override:
                        self._exit_stack.enter_context(
                            path_override_context({"root_dir": self._root_dir})
                        )

                # Create and enter resource manager using ExitStack
                logger.debug("Creating resource manager")
                resource_manager = MLflowResourceManager(self._mlflow_config)
                self._resource_manager = self._exit_stack.enter_context(resource_manager)

                # Get server info if available
                server_info = self._resource_manager.get_server_info()
                server_url = getattr(server_info, "url", None) if server_info else None

                if server_url:
                    self._server_url = server_url
                elif self._configured_server_url:
                    self._server_url = self._configured_server_url

                # Skip redundant health check - server was already validated during startup
                # Performing another check here can interfere with server initialization
                self._server_status = {"running": True, "response_time": None} if server_url else None

                logger.info(f"MLflow resources initialized - Server: {server_url}")

            except Exception as e:
                logger.error(f"Failed to initialize MLflow resources: {e}")
                # Cleanup via ExitStack
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                self._resource_manager = None
                self._server_url = None
                self._server_status = None
                raise
        else:
            logger.debug("Skipping resource initialization - no config provided")

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Context manager exit with cleanup via ExitStack.

        Ensures all MLflow resources (server, client, runs) are properly cleaned up,
        even if exceptions occurred during execution.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_val: Exception instance if an exception occurred, None otherwise.
            exc_tb: Exception traceback if an exception occurred, None otherwise.

        Example:
            ```python
            with tracker:
                # Even if exception occurs here
                raise ValueError("Something went wrong")
            # Cleanup still happens
            ```
        """
        logger.info(f"MLflowTracker.__exit__ called - exc_type: {exc_type}, exc_val: {exc_val}")
        if self._exit_stack:
            try:
                logger.info("MLflowTracker: Cleaning up MLflow resources via ExitStack")
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                logger.info("MLflowTracker: ExitStack cleanup completed")
            except Exception as e:
                logger.warning(f"Failed to cleanup MLflow resources: {e}")
            finally:
                self._exit_stack = None
                self._resource_manager = None
        self._server_url = None
        self._server_status = None
        self._mlflow_config = None
        logger.info("MLflowTracker.__exit__ completed")

    @contextmanager
    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
    ) -> AbstractContextManager[IRunContext]:
        """Create MLflow run using resource manager.

        Creates a new MLflow run within the specified experiment. If the experiment
        doesn't exist, it will be created automatically. Supports nested runs for
        hierarchical tracking (e.g., parent optimization run with child trial runs).

        Args:
            experiment_name: Name of experiment to create run under. If None, uses
                default experiment ("DLKit").
            run_name: Optional name for this specific run. If None, MLflow generates
                a unique name automatically.
            nested: If True, creates a child run under the currently active parent run.
                Requires an active parent run context.

        Yields:
            IRunContext: Run context for logging metrics, params, and artifacts.

        Raises:
            RuntimeError: If MLflow not configured (setup_mlflow_config not called).

        Example:
            ```python
            tracker = MLflowTracker()
            tracker.setup_mlflow_config(settings.MLFLOW)

            with tracker:
                # Basic run
                with tracker.create_run(experiment_name="training") as run:
                    run.log_metrics({"loss": 0.5})

                # Nested runs for hyperparameter search
                with tracker.create_run(experiment_name="hp_opt") as parent:
                    for trial in trials:
                        with tracker.create_run(
                            experiment_name="hp_opt",
                            run_name=f"trial_{trial.id}",
                            nested=True
                        ) as child:
                            child.log_params(trial.params)
                            child.log_metrics({"score": trial.score})
            ```
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call setup_mlflow_config first")

        exp_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT

        with self._resource_manager.create_run(
            experiment_name=exp_name,
            run_name=run_name,
            nested=nested,
        ) as run_context:
            yield run_context

    def log_settings(self, settings: GeneralSettings, run_context: IRunContext) -> None:
        """Save complete configuration settings as MLflow TOML artifact.

        Creates a temporary TOML file from settings (excluding unset values) and
        uploads it as an artifact to the run. This enables full reproducibility -
        you can recreate the exact training environment from the artifact.

        Args:
            settings: Complete settings object to serialize and log.
            run_context: Active run context to log artifact to.

        Raises:
            RuntimeError: If serialization or artifact logging fails.

        Example:
            ```python
            with tracker.create_run("training") as run:
                # Log full config for reproducibility
                tracker.log_settings(settings, run)
                # Creates "GeneralSettings.toml" artifact in run
            ```
        """
        try:
            # Log full config as TOML artifact (excluding unset values)
            from dlkit.tools.io.config import write_config
            from pathlib import Path
            import os
            import tempfile

            # Create temporary TOML file with unset values excluded
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as temp_file:
                write_config(settings, temp_file.name, exclude_unset=True)
                temp_path = Path(temp_file.name)

            try:
                # Log TOML file as artifact only using run context
                run_context.log_artifact(temp_path, artifact_dir="")
            finally:
                # Clean up temp file
                if temp_path.exists():
                    os.unlink(temp_path)

        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_model_parameters(
        self, model: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Log model hyperparameters extracted from settings.MODEL.

        Extracts hyperparameters from settings.MODEL (if present) and logs them as
        run parameters. Automatically excludes structural fields (name, module_path,
        checkpoint, shape) to log only actual hyperparameters.

        Args:
            model: Model instance (currently ignored - may be used in future for
                runtime parameter extraction).
            run_context: Active run context to log parameters to.
            settings: Settings object containing MODEL configuration.

        Raises:
            RuntimeError: If parameter extraction or logging fails.

        Example:
            ```python
            # settings.MODEL contains:
            # ModelComponent(name="cnn", hidden_dim=256, num_layers=3, dropout=0.1)

            with tracker.create_run("training") as run:
                tracker.log_model_parameters(model, run, settings)
                # Logs: {"hidden_dim": 256, "num_layers": 3, "dropout": 0.1}
                # Excludes: "name" (structural field)
            ```
        """
        try:
            if settings.MODEL is None:
                return

            # Use settings.MODEL and dump its hyperparameters - so simple!
            params = settings.MODEL.model_dump(exclude_none=True)

            # Remove component-specific fields that aren't hyperparameters
            component_fields = {"name", "module_path", "checkpoint", "shape"}
            hparams = {k: v for k, v in params.items() if k not in component_fields}

            # Only log if we have hyperparameters to log
            if hparams:
                run_context.log_params(hparams)

        except Exception as e:
            raise RuntimeError("Couldn't log settings") from e

    def log_dataset_to_run(
        self, datamodule: Any, run_context: IRunContext, settings: GeneralSettings
    ) -> None:
        """Log dataset to MLflow run.

        Converts DLKit datasets to MLflow dataset format and logs them for reproducibility.
        Currently supports FlexibleDataset with NumPy conversion. Other dataset types
        are logged as warnings.

        Args:
            datamodule: Lightning DataModule containing the dataset
            run_context: Active run context to log to
            settings: Settings containing dataset configuration

        Example:
            ```python
            with tracker.create_run("training") as run:
                tracker.log_dataset_to_run(datamodule, run, settings)
                # Logs dataset with features and targets to MLflow
            ```
        """
        try:
            # Extract dataset from datamodule
            dataset = getattr(datamodule, "dataset", None)
            if dataset is None:
                logger.debug("No dataset found in datamodule, skipping dataset logging")
                return

            # Import here to avoid circular dependencies
            from dlkit.core.datasets.flexible import FlexibleDataset
            import mlflow.data

            # Type-based mapping: currently only NumPy (FlexibleDataset)
            if isinstance(dataset, FlexibleDataset):
                # Convert torch tensors to numpy
                features_np = {k: v.cpu().numpy() for k, v in dataset.features.items()}
                targets_np = {k: v.cpu().numpy() for k, v in dataset.targets.items()} if dataset.targets else None

                # Extract dataset name and source
                dataset_name = getattr(settings.DATASET, "name", None) or "training_data"

                # Try to extract source path from dataset features (first feature path if available)
                dataset_source = None
                try:
                    ds_settings = settings.DATASET
                    if hasattr(ds_settings, "features") and ds_settings.features:
                        # Get first feature's path
                        first_feature = next(iter(ds_settings.features), None)
                        if first_feature and hasattr(first_feature, "path"):
                            dataset_source = str(first_feature.path)
                except Exception:
                    pass

                # Create MLflow NumpyDataset
                mlflow_dataset = mlflow.data.from_numpy(
                    features=features_np,
                    targets=targets_np,
                    name=dataset_name,
                    source=dataset_source
                )

                # Prepare tags with dataset metadata
                tags = {}
                try:
                    split_cfg = settings.DATASET.split
                    tags["split_test_ratio"] = str(split_cfg.test_ratio)
                    tags["split_val_ratio"] = str(split_cfg.val_ratio)
                except Exception:
                    pass

                # Add dataset type from meta if available
                try:
                    # Access from BuildComponents.meta would require passing it
                    # For now, infer from settings
                    dataset_type = getattr(settings.DATASET, "type", None)
                    if dataset_type:
                        tags["dataset_type"] = str(dataset_type)
                except Exception:
                    pass

                # Log the dataset
                run_context.log_dataset(mlflow_dataset, context="training", tags=tags if tags else None)
                logger.info(f"Logged dataset '{dataset_name}' to MLflow")

            else:
                # Other dataset types not yet supported
                dataset_class_name = type(dataset).__name__
                logger.warning(
                    f"Dataset type '{dataset_class_name}' not yet supported for MLflow logging. "
                    f"Only FlexibleDataset (NumPy) is currently supported."
                )

        except Exception as e:
            # Don't fail the run if dataset logging fails
            logger.warning(f"Failed to log dataset to MLflow: {e}")

    def setup_mlflow_config(
        self,
        mlflow_config: Any,
        *,
        root_dir: Path | str | None = None,
    ) -> tuple[str | None, dict | None]:
        """Configure MLflow tracking - stores config for deferred resource initialization.

        Stores the MLflow configuration for later use. Actual resource initialization
        (server startup, client creation) is deferred until __enter__() to follow
        proper context manager protocol and ensure cleanup.

        Args:
            mlflow_config: MLflow configuration settings (typically from settings.MLFLOW).
                Must have 'enabled' attribute. If enabled=False, tracking is skipped.

        Returns:
            tuple[str | None, dict | None]: Tuple of (configured_server_url, server_status).
                Always returns (url, None) or (None, None) since resources aren't
                initialized yet. The URL is extracted from config but server isn't started.

        Example:
            ```python
            from dlkit.tools.config import GeneralSettings

            settings = GeneralSettings.from_toml("config.toml")
            tracker = MLflowTracker()

            # Just stores config - no server startup yet
            url, status = tracker.setup_mlflow_config(settings.MLFLOW)
            print(f"Will use server: {url}")  # May be None

            # Resources initialized here
            with tracker:
                with tracker.create_run("exp") as run:
                    run.log_metrics({"loss": 0.5})
            ```
        """
        self._root_dir = self._determine_root_dir(root_dir)
        self._mlflow_config = self._normalize_mlflow_config(mlflow_config, self._root_dir)
        self._server_status = None
        self._server_url = None
        self._configured_server_url = None

        # Skip setup if disabled
        if (
            self.disable_autostart
            or not mlflow_config
            or not getattr(mlflow_config, "enabled", False)
        ):
            logger.debug("MLflow disabled or autostart disabled")
            return None, None

        configured_url = None
        client = getattr(mlflow_config, "client", None)
        if client:
            tracking_uri = getattr(client, "tracking_uri", None)
            if tracking_uri:
                tracking_uri_str = str(tracking_uri)
                if tracking_uri_str.startswith(("http://", "https://")):
                    configured_url = tracking_uri_str

        if configured_url is None:
            configured_url = self._derive_server_url(mlflow_config)

        self._configured_server_url = configured_url

        logger.debug("MLflow config stored - will initialize in context entry")
        return configured_url, None

    def _determine_root_dir(self, candidate: Path | str | None) -> Path | None:
        if candidate is not None:
            return Path(candidate).resolve()

        ctx = get_current_path_context()
        if ctx and getattr(ctx, "root_dir", None):
            try:
                return Path(ctx.root_dir).resolve()
            except TypeError:
                return Path(str(ctx.root_dir)).resolve()

        return None

    def _normalize_mlflow_config(
        self, mlflow_config: Any, root_dir: Path | None
    ) -> Any:
        if not mlflow_config or root_dir is None:
            return mlflow_config

        try:
            server = getattr(mlflow_config, "server", None)
            if server is not None:
                normalized_server = self._normalize_server_paths(server, root_dir)
                if normalized_server is not server:
                    mlflow_config = mlflow_config.model_copy(update={"server": normalized_server})
        except Exception:
            return mlflow_config

        return mlflow_config

    @staticmethod
    def _normalize_server_paths(server_config: Any, root_dir: Path) -> Any:
        updates: dict[str, Any] = {}

        backend_uri = getattr(server_config, "backend_store_uri", None)
        if backend_uri:
            backend_str = str(backend_uri)
            normalized_backend = MLflowTracker._normalize_backend_uri(backend_str, root_dir)
            if normalized_backend != backend_str:
                updates["backend_store_uri"] = normalized_backend

        artifacts_dest = getattr(server_config, "artifacts_destination", None)
        if artifacts_dest:
            artifacts_str = str(artifacts_dest)
            normalized_artifacts = MLflowTracker._normalize_artifacts_destination(
                artifacts_str, root_dir
            )
            if normalized_artifacts != artifacts_str:
                updates["artifacts_destination"] = normalized_artifacts

        if updates:
            return server_config.model_copy(update=updates)
        return server_config

    @staticmethod
    def _normalize_backend_uri(uri: str, root_dir: Path) -> str:
        from urllib.parse import urlparse

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
        from urllib.parse import urlparse

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

    def _derive_server_url(self, mlflow_config: Any) -> str | None:
        """Derive server URL from resource manager or config.

        Args:
            mlflow_config: MLflow configuration

        Returns:
            Server URL if available
        """
        # Try to get URL from resource manager first
        if self._resource_manager:
            server_info = self._resource_manager.get_server_info()
            if server_info:
                return getattr(server_info, "url", None)

        client = getattr(mlflow_config, "client", None) if mlflow_config else None
        if client:
            tracking_uri = getattr(client, "tracking_uri", None)
            if tracking_uri:
                tracking_uri_str = str(tracking_uri)
                if tracking_uri_str.startswith(("http://", "https://")):
                    return tracking_uri_str

        return None

    def cleanup_server(self) -> None:
        """Clean up MLflow resources.

        Note: This method exists for backward compatibility. Prefer using
        the context manager protocol (with statement) instead.
        """
        if self._exit_stack:
            logger.debug("Cleaning up via ExitStack")
            try:
                self._exit_stack.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Failed to cleanup: {e}")
            finally:
                self._exit_stack = None
                self._resource_manager = None

    def _health_check(self, server_url: str) -> dict | None:
        """Check MLflow server health."""
        try:
            from dlkit.interfaces.servers.health_checker import HTTPHealthChecker

            status = HTTPHealthChecker(request_timeout=HEALTHCHECK_TIMEOUT_S).check_health(
                server_url
            )
            return {
                "running": bool(getattr(status, "is_running", False)),
                "response_time": getattr(status, "response_time", None),
                "error": getattr(status, "error_message", None),
            }
        except Exception:
            return None

    def get_server_url(self) -> str | None:
        """Return the best-known server URL, if any."""
        return self._server_url or self._configured_server_url

    def get_server_status(self, server_url: str | None = None) -> dict | None:
        """Get the most recent server status information."""
        if self._server_status is not None:
            return dict(self._server_status)

        # If we have a server URL but no cached status, attempt a lightweight check
        url = server_url or self.get_server_url()
        if url and not self.skip_health_checks:
            status = self._health_check(url)
            if status is not None:
                self._server_status = status
                return dict(status)
        return None
