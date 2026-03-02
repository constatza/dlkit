"""Composable tracking decorator following OCP and SRP.

Coordinates experiment tracking lifecycle by composing focused services.
Each service has a single responsibility following SOLID principles.
"""

from __future__ import annotations

from dlkit.interfaces.api.domain import TrainingResult
from dlkit.runtime.workflows.factories.build_factory import BuildComponents
from dlkit.tools.config import GeneralSettings
from dlkit.tools.utils.error_handling import raise_error
from dlkit.tools.utils.logging_config import get_logger

from ..core.interfaces import ITrainingExecutor
from .artifact_logger import ArtifactLogger
from .config_accessor import ConfigAccessor
from .interfaces import IExperimentTracker, IRunContext
from .metric_logger import MetricLogger
from .result_enricher import ResultEnricher

logger = get_logger(__name__)


class TrackingDecorator(ITrainingExecutor):
    """Composable experiment tracking decorator following OCP, DIP, and SRP.

    Coordinates tracking lifecycle by composing focused services:
    - ConfigAccessor: Type-safe configuration access
    - MetricLogger: Metric logging to MLflow
    - ArtifactLogger: Checkpoint and artifact logging
    - ResultEnricher: Result enrichment with tracking metadata

    Each service has a single responsibility and can be tested independently.
    Resource management is delegated to the tracker implementation.

    Args:
        executor: Core training executor to decorate
        tracker: Experiment tracker implementation
        metric_logger: Metric logging service (optional, created if not provided)
        artifact_logger: Artifact logging service (optional, created if not provided)
        result_enricher: Result enrichment service (optional, created if not provided)
    """

    def __init__(
        self,
        executor: ITrainingExecutor,
        tracker: IExperimentTracker,
        metric_logger: MetricLogger | None = None,
        artifact_logger: ArtifactLogger | None = None,
        result_enricher: ResultEnricher | None = None,
    ):
        """Initialize with base executor, tracker, and optional services.

        Services are created lazily if not provided, enabling dependency injection
        for testing while maintaining simple usage for production.

        Args:
            executor: Core training executor to decorate
            tracker: Experiment tracker implementation
            metric_logger: Metric logging service (optional)
            artifact_logger: Artifact logging service (optional)
            result_enricher: Result enrichment service (optional)
        """
        self._executor = executor
        self._tracker = tracker
        self._metric_logger = metric_logger or MetricLogger(tracker)
        self._artifact_logger = artifact_logger or ArtifactLogger(tracker)
        self._result_enricher = result_enricher or ResultEnricher(tracker)

    def execute(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
    ) -> TrainingResult:
        """Execute training with experiment tracking.

        Coordinates the tracking lifecycle:
        1. Setup tracking configuration
        2. Enter tracker context (resource management)
        3. Create MLflow run
        4. Execute training with metric logging
        5. Log artifacts and enrich result
        6. Exit tracker context (cleanup)

        Args:
            components: Pre-built training components
            settings: Global training settings

        Returns:
            TrainingResult enriched with tracking metadata

        Raises:
            WorkflowError: If training or tracking fails
        """
        # Configure the tracker before entering its context
        logger.debug("Setting up tracking")
        server_url, server_status = self._setup_tracking(settings)

        # Use tracker as context manager for proper resource management
        logger.info("Starting training with MLflow tracking")
        with self._tracker:  # type: ignore[attr-defined]
            logger.debug("Entering tracker context, resolving metadata")
            server_url, server_status = self._resolve_tracker_metadata(server_url, server_status)
            logger.debug("Executing training with tracking")
            result = self._execute_with_tracking(components, settings, server_url, server_status)
            logger.debug("Training completed, exiting tracker context")
            return result

    def _execute_with_tracking(  # noqa: PLR0911
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        server_url: str | None = None,
        server_status: dict | None = None,
    ) -> TrainingResult:
        """Execute training with tracking - separated for SRP compliance.

        Args:
            components: Pre-built training components
            settings: Global training settings
            server_url: MLflow server URL if available
            server_status: MLflow server status if available

        Returns:
            TrainingResult enriched with tracking metadata

        Raises:
            WorkflowError: If training or tracking fails
        """
        try:  # pyright: ignore[reportReturnType]
            # Get run configuration
            logger.debug("Extracting run config")
            run_config = self._extract_run_config(settings)

            # Execute training within tracking context
            logger.debug(f"Creating MLflow run with config: {run_config}")
            pending_registration = None
            enriched_result = None
            with self._tracker.create_run(**run_config) as run_context:
                logger.debug("MLflow run created successfully")

                # Log metadata and configuration
                self._log_tracking_metadata(run_context, server_url, server_status)
                self._log_configuration(components, settings, run_context)

                # Inject MLflow logger into trainer for automatic metric logging
                self._inject_mlflow_logger(components, run_context, settings)

                # Execute core training
                result = self._executor.execute(components, settings)

                # Log final summary metrics (delegate to MetricLogger)
                self._metric_logger.log_summary_metrics(result, run_context)

                # Log training artifacts (delegate to ArtifactLogger)
                pending_registration = self._artifact_logger.log_training_artifacts(
                    components, settings, run_context
                )

                # Log user-defined custom artifacts and params (delegate to ArtifactLogger)
                self._artifact_logger.log_user_artifacts(settings, run_context, result)

                # Enrich result with tracking metadata (delegate to ResultEnricher)
                enriched_result = self._result_enricher.enrich_result(
                    result, settings, server_url, server_status
                )

            self._artifact_logger.finalize_model_registration(pending_registration, run_context)
            if enriched_result is None:
                raise RuntimeError("Tracking result enrichment failed unexpectedly")
            return enriched_result

        except Exception as e:
            logger.error(f"TrackingDecorator: Exception caught: {type(e).__name__}: {e}")
            raise_error("Training with tracking failed", e, stage="tracking")  # type: ignore[return-value]

    def _setup_tracking(
        self,
        settings: GeneralSettings,
    ) -> tuple[str | None, dict | None]:
        """Setup tracking configuration.

        Delegates to tracker's setup method if available, using ConfigAccessor
        for type-safe settings access.

        Args:
            settings: Global settings

        Returns:
            Tuple of (server_url, server_status)
        """
        accessor = ConfigAccessor(settings)
        mlflow_config = accessor.get_mlflow_config()

        if hasattr(self._tracker, "setup_mlflow_config"):
            root_dir = accessor.get_session_root_dir()
            setup_fn = getattr(self._tracker, "setup_mlflow_config")
            try:
                return setup_fn(mlflow_config, root_dir=root_dir)
            except TypeError as exc:
                # Fall back for trackers that have not yet adopted the root_dir kwarg
                if "unexpected keyword argument 'root_dir'" in str(exc):
                    logger.debug("Tracker does not support root_dir parameter, using fallback")
                    return setup_fn(mlflow_config)
                raise

        return None, None

    def _resolve_tracker_metadata(
        self,
        server_url: str | None,
        server_status: dict | None,
    ) -> tuple[str | None, dict | None]:
        """Merge tracker-provided metadata discovered during context entry.

        Gives tracker a chance to provide metadata after entering its context.
        Metadata from setup is preferred if already provided.

        Args:
            server_url: Server URL from setup
            server_status: Server status from setup

        Returns:
            Tuple of (resolved_url, resolved_status)
        """
        # Prefer metadata returned from setup if provided
        if server_url is None and hasattr(self._tracker, "get_server_url"):
            try:
                candidate_url = self._tracker.get_server_url()  # type: ignore[attr-defined]
            except TypeError:
                # Some implementations may require server_url parameter
                candidate_url = self._tracker.get_server_url(server_url)  # type: ignore[attr-defined]
            if candidate_url:
                server_url = candidate_url

        if server_status is None and hasattr(self._tracker, "get_server_status"):
            try:
                candidate_status = self._tracker.get_server_status(server_url)  # type: ignore[attr-defined]
            except TypeError:
                # Some implementations may not require server_url parameter
                candidate_status = self._tracker.get_server_status()  # type: ignore[attr-defined]
            if candidate_status is not None:
                server_status = candidate_status

        return server_url, server_status

    def _extract_run_config(self, settings: GeneralSettings) -> dict:
        """Extract run configuration from settings.

        Uses ConfigAccessor for type-safe configuration access and
        delegates experiment naming to naming module.

        Args:
            settings: Global settings

        Returns:
            Run configuration dictionary for tracker.create_run()
        """
        from .naming import determine_experiment_name

        accessor = ConfigAccessor(settings)
        mlflow_config = accessor.get_mlflow_config()

        return {
            "experiment_name": determine_experiment_name(settings, mlflow_config),
            "run_name": accessor.get_run_name(),
            "nested": self._should_use_nested_runs(),
        }

    def _log_tracking_metadata(
        self,
        run_context: IRunContext,
        server_url: str | None,
        server_status: dict | None,
    ) -> None:
        """Log tracking metadata to run context.

        Logs server URL and status information as MLflow tags.

        Args:
            run_context: Run context for logging
            server_url: Server URL if available
            server_status: Server status if available
        """
        if server_url:
            run_context.set_tag("mlflow_server_url", server_url)

        if server_status is not None:
            run_context.set_tag("mlflow_server_running", str(bool(server_status.get("running"))))
            run_context.set_tag(
                "mlflow_server_response_time", str(server_status.get("response_time"))
            )

    def _log_configuration(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext,
    ) -> None:
        """Log configuration and model parameters.

        Delegates actual logging to tracker implementation.

        Args:
            components: Build components
            settings: Global settings
            run_context: Run context for logging
        """
        # Log configuration
        self._tracker.log_settings(settings, run_context)

        # Log dataset
        self._log_dataset_to_run(components, settings, run_context)

        # Log model hyperparameters
        self._tracker.log_model_parameters(components.model, run_context, settings)

    def _log_dataset_to_run(
        self,
        components: BuildComponents,
        settings: GeneralSettings,
        run_context: IRunContext,
    ) -> None:
        """Log dataset to MLflow run.

        Delegates to tracker if it supports dataset logging.

        Args:
            components: Build components containing datamodule
            settings: Global settings
            run_context: Run context for logging
        """
        try:
            if hasattr(self._tracker, "log_dataset_to_run"):
                self._tracker.log_dataset_to_run(components.datamodule, run_context, settings)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Failed to log dataset: {e}")

    def _inject_mlflow_logger(
        self,
        components: BuildComponents,
        run_context: IRunContext,
        settings: GeneralSettings,
    ) -> None:
        """Inject MLflow epoch logger callback for metric logging with epochs as x-axis.

        Creates and adds MLflowEpochLogger callback to trainer for automatic
        metric logging during training.

        Args:
            components: Build components
            run_context: Run context with client and run_id
            settings: Global settings
        """
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                logger.debug("No trainer found in components")
                return

            # Create callback that logs metrics with epoch numbers instead of steps
            from dlkit.core.training.callbacks import MLflowEpochLogger

            epoch_logger = MLflowEpochLogger(run_context)

            # Add callback to trainer
            if not hasattr(trainer, "callbacks"):
                trainer.callbacks = []
            trainer.callbacks.append(epoch_logger)
            logger.debug(f"Injected MLflowEpochLogger callback (run_id={run_context.run_id})")

        except Exception as e:
            logger.warning(f"Failed to inject MLflow epoch logger: {e}")

    def _should_use_nested_runs(self) -> bool:
        """Determine if nested runs should be used based on existing MLflow run context.

        Returns:
            True if there's already an active MLflow run (nested run required)
        """
        try:
            import mlflow

            # Check if there's already an active MLflow run
            active_run = mlflow.active_run()
            return active_run is not None
        except Exception as e:
            logger.warning(f"Failed to check for active MLflow run: {e}")
            return False
