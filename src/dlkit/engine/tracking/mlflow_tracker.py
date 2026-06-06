"""MLflow adapter implementing tracking abstractions."""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType

from dlkit.engine.tracking.interfaces import IExperimentTracker, IRunContext
from dlkit.infrastructure.config.mlflow_settings import MLflowSettings
from dlkit.infrastructure.utils.logging_config import get_logger

from .backend import LocalSqliteBackend, TrackingBackend, select_backend
from .mlflow_resource_manager import MLflowResourceManager

logger = get_logger(__name__)

# MLflow integration constants
MLFLOW_DEFAULT_EXPERIMENT = "DLKit"


class MLflowTracker(IExperimentTracker):
    """MLflow implementation of experiment tracker using resource manager pattern.

    Provides MLflow-based experiment tracking with proper resource lifecycle management
    through MLflowResourceManager. Handles client creation, experiment/run management,
    and guaranteed cleanup.

    The tracker should be used as a context manager to ensure proper resource cleanup:

    Example:
        ```python
        from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker()
        tracker.configure(settings.MLFLOW)

        with tracker:  # Initializes resources
            with tracker.create_run(experiment_name="training") as run:
                run.log_params({"learning_rate": 0.001})
                run.log_metrics({"loss": 0.5}, step=1)
                tracker.log_settings(settings, run)
        # Resources cleaned up automatically
        ```

    Attributes:
        disable_autostart (bool): Skip automatic tracker setup.
    """

    def __init__(
        self,
        disable_autostart: bool = False,
        probe: Callable[[], bool] | None = None,
    ):
        """Initialize MLflow tracker.

        Args:
            disable_autostart: If True, skip automatic tracker setup.
            probe: Optional callable to detect a local MLflow server.
                Defaults to ``local_host_alive`` from ``uri_resolver``.
        """
        self.disable_autostart = disable_autostart
        self._probe = probe
        self._resource_manager: MLflowResourceManager | None = None
        self._mlflow_config: MLflowSettings | None = None
        self._exit_stack: ExitStack | None = None
        self._backend: TrackingBackend | None = None

    def __enter__(self) -> MLflowTracker:
        """Context manager entry - initializes MLflow resources using ExitStack.

        Creates and enters the MLflowResourceManager which handles client
        initialization and experiment setup. Uses ExitStack for nested context
        management to ensure cleanup.

        Returns:
            MLflowTracker: Self for context manager protocol.

        Raises:
            Exception: If resource initialization fails.
        """
        logger.debug("MLflow tracker entering (configured={})", self._mlflow_config is not None)

        if self._mlflow_config and not self.disable_autostart:
            try:
                self._exit_stack = ExitStack()
                self._exit_stack.__enter__()

                logger.debug("Selecting tracking backend")
                self._backend = select_backend(probe=self._probe)

                logger.debug("Creating resource manager")
                resource_manager = MLflowResourceManager(self._mlflow_config, self._backend)
                self._resource_manager = self._exit_stack.enter_context(resource_manager)

                logger.debug("MLflow resources initialized")

            except Exception as e:
                logger.error("Failed to initialize MLflow resources: {}", e)
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                self._resource_manager = None
                self._backend = None
                raise
        else:
            logger.debug(
                "Skipping resource initialization - no config provided or autostart disabled"
            )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit with cleanup via ExitStack."""
        logger.debug("MLflowTracker.__exit__ called - exc_type: {}, exc_val: {}", exc_type, exc_val)
        if self._exit_stack:
            try:
                logger.debug("MLflowTracker: Cleaning up MLflow resources via ExitStack")
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                logger.debug("MLflowTracker: ExitStack cleanup completed")
            except Exception as e:
                logger.warning("Failed to clean up MLflow resources: {}", e)
            finally:
                self._exit_stack = None
                self._resource_manager = None
                self._backend = None
        self._mlflow_config = None
        logger.debug("MLflowTracker.__exit__ completed")

    def create_run(
        self,
        experiment_name: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> AbstractContextManager[IRunContext]:
        """Create MLflow run using resource manager.

        Args:
            experiment_name: Name of experiment to create run under.
            run_name: Optional name for this specific run.
            nested: If True, creates a child run under the currently active parent run.
            tags: Optional tags to attach to the run.

        Returns:
            AbstractContextManager[IRunContext]: Context manager yielding active run context.

        Raises:
            RuntimeError: If MLflow not configured (configure not called).
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call configure() before entering context")

        exp_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT

        return self._resource_manager.create_run(
            experiment_name=exp_name,
            run_name=run_name,
            nested=nested,
            tags=tags,
        )

    def get_tracking_uri(self) -> str | None:
        """Return the resolved tracking URI, or None if not initialized.

        Returns:
            Tracking URI string or None.
        """
        return self._backend.tracking_uri() if self._backend is not None else None

    def is_local(self) -> bool:
        """Return True when using a local SQLite backend.

        Returns:
            True if the backend is ``LocalSqliteBackend``.
        """
        return isinstance(self._backend, LocalSqliteBackend)

    def has_active_parent_run(self) -> bool:
        """Report whether an active parent run already exists for nesting."""
        if self._resource_manager is None:
            return False
        return self._resource_manager.has_active_parent_run()

    def configure(self, config: MLflowSettings) -> None:
        """Store MLflow config with no side effects."""
        self._mlflow_config = config
        logger.debug("MLflow config stored - will initialize in context entry")
