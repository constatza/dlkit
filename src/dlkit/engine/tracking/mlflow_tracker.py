"""MLflow adapter implementing tracking abstractions."""

from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from types import TracebackType

from dlkit.engine.tracking.interfaces import IExperimentTracker, IRunContext
from dlkit.infrastructure.config.tracking_settings import TrackingSettings
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
        tracker.configure(settings.tracking)

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
        self._mlflow_config: TrackingSettings | None = None
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
                self._backend = select_backend(
                    uri=self._mlflow_config.uri,
                    probe=self._probe,
                )

                logger.debug("Creating resource manager")
                resource_manager = MLflowResourceManager(self._mlflow_config, self._backend)
                self._resource_manager = self._exit_stack.enter_context(resource_manager)

                logger.debug("MLflow resources initialized")

            except Exception:
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

        Delegates to the resource manager once one exists, since its backend
        is the *effective* one — it may have changed from the one selected
        in `__enter__` if connectivity failed and fell back to local
        tracking. Falls back to the pre-entry `_backend` only when no
        resource manager has been created yet.

        Returns:
            Tracking URI string or None.
        """
        if self._resource_manager is not None:
            return self._resource_manager.get_tracking_uri()
        return self._backend.tracking_uri() if self._backend is not None else None

    def is_local(self) -> bool:
        """Return True when using a local SQLite backend.

        See `get_tracking_uri` for why this delegates to the resource
        manager's effective backend rather than the pre-entry one.

        Returns:
            True if the effective backend is ``LocalSqliteBackend``.
        """
        if self._resource_manager is not None:
            return isinstance(self._resource_manager.backend, LocalSqliteBackend)
        return isinstance(self._backend, LocalSqliteBackend)

    def is_active(self) -> bool:
        """Return True — MLflow tracker performs real experiment tracking.

        Returns:
            Always True for this backend.
        """
        return True

    def has_active_parent_run(self) -> bool:
        """Report whether an active parent run already exists for nesting."""
        if self._resource_manager is None:
            return False
        return self._resource_manager.has_active_parent_run()

    def set_run_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag on a run by id, without requiring it to be the active run.

        Unlike ``IRunContext.set_tag``, which only tags the run its context
        wraps, this reaches an arbitrary run — active, already closed, or
        never opened by this tracker instance — via a fresh client handle.
        Used for post-hoc tagging, e.g. a multirun orchestrator tagging a
        child run with its sweep's parent run id after the child's own
        train()/optimize()/converge() call has already opened and closed
        that run independently.

        Args:
            run_id: MLflow run id to tag.
            key: Tag name/key.
            value: Tag value.

        Raises:
            RuntimeError: If MLflow not configured (configure not called).
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call configure() before entering context")
        self._resource_manager.get_client().set_tag(run_id, key, value)

    def get_run_context(self, run_id: str) -> IRunContext:
        """Return a post-hoc run context for an existing run, without activating it.

        Unlike ``create_run()``, this never calls ``mlflow.start_run()`` and
        never touches MLflow's global active-run state — it wraps ``run_id``
        in a client-backed run context that logs via direct ``MlflowClient``
        calls. Safe to use on a run that's already finished (e.g. a multirun
        sweep's own parent run, deliberately closed before children run so
        each child's own top-level run doesn't collide with it) or on any
        run this tracker instance didn't itself open.

        Args:
            run_id: MLflow run id to wrap.

        Returns:
            IRunContext: A run context backed by run_id.

        Raises:
            RuntimeError: If MLflow not configured (configure not called).
        """
        if not self._resource_manager:
            raise RuntimeError("MLflow not configured - call configure() before entering context")
        from .mlflow_run_context import ClientBasedRunContext

        client = self._resource_manager.get_client()
        experiment_id = client.get_run(run_id).info.experiment_id
        return ClientBasedRunContext(
            client,
            run_id,
            tracking_uri=self.get_tracking_uri() or "",
            experiment_id=experiment_id,
        )

    def configure(self, config: TrackingSettings) -> None:
        """Store tracking config with no side effects."""
        self._mlflow_config = config
        logger.debug("Tracking config stored - will initialize in context entry")
