"""Composable tracking decorator following OCP and SRP.

Coordinates experiment tracking lifecycle by composing focused services.
Each service has a single responsibility following SOLID principles.
"""

from __future__ import annotations

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.tracking.artifact_logger import ArtifactLogger
from dlkit.engine.tracking.config_accessor import ConfigAccessor
from dlkit.engine.tracking.interfaces import (
    IDatasetLogger,
    IExperimentTracker,
    IRunContext,
    ITrackingSetup,
)
from dlkit.engine.tracking.metric_logger import MetricLogger
from dlkit.engine.tracking.result_enricher import ResultEnricher
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.interfaces import ITrainingExecutor
from dlkit.infrastructure.config import GeneralSettings
from dlkit.infrastructure.config.workflow_configs import (
    OptimizationWorkflowConfig,
    TrainingWorkflowConfig,
)
from dlkit.infrastructure.utils.error_handling import raise_error
from dlkit.infrastructure.utils.logging_config import get_logger

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
        hooks: Optional functional extension points for lifecycle events
        metric_logger: Metric logging service (optional, created if not provided)
        artifact_logger: Artifact logging service (optional, created if not provided)
        result_enricher: Result enrichment service (optional, created if not provided)
    """

    def __init__(
        self,
        executor: ITrainingExecutor,
        tracker: IExperimentTracker,
        hooks: LifecycleHooks | None = None,
        metric_logger: MetricLogger | None = None,
        artifact_logger: ArtifactLogger | None = None,
        result_enricher: ResultEnricher | None = None,
    ):
        """Initialize with base executor, tracker, optional hooks, and optional services.

        Args:
            executor: Core training executor to decorate
            tracker: Experiment tracker implementation
            hooks: Optional functional extension points for lifecycle events
            metric_logger: Metric logging service (optional)
            artifact_logger: Artifact logging service (optional)
            result_enricher: Result enrichment service (optional)
        """
        self._executor = executor
        self._tracker = tracker
        self._hooks = hooks
        self._metric_logger = metric_logger or MetricLogger(tracker)
        self._artifact_logger = artifact_logger or ArtifactLogger(tracker)
        self._result_enricher = result_enricher or ResultEnricher(tracker)

    def execute(
        self,
        components: RuntimeComponents,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
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
        self._setup_tracking(settings)

        # Use tracker as context manager for proper resource management
        logger.info("Starting training with MLflow tracking")
        with self._tracker:
            # Get tracking URI after __enter__ activates the backend
            tracking_uri = (
                self._tracker.get_tracking_uri()
                if hasattr(self._tracker, "get_tracking_uri")
                else None
            )
            is_local = self._tracker.is_local() if hasattr(self._tracker, "is_local") else False
            if tracking_uri:
                if is_local:
                    logger.info("Using local file-backed MLflow tracking URI: {}", tracking_uri)
                else:
                    logger.info("Using MLflow server tracking URI: {}", tracking_uri)
            logger.debug("Executing training with tracking")
            result = self._execute_with_tracking(components, settings, tracking_uri)
            logger.debug("Training completed, exiting tracker context")
            return result

    def _execute_with_tracking(
        self,
        components: RuntimeComponents,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        tracking_uri: str | None = None,
    ) -> TrainingResult:
        """Execute training with tracking - separated for SRP compliance.

        Args:
            components: Pre-built training components
            settings: Global training settings
            tracking_uri: Resolved MLflow tracking URI if available

        Returns:
            TrainingResult enriched with tracking metadata

        Raises:
            WorkflowError: If training or tracking fails
        """
        try:
            # Get run configuration (includes merged tags)
            logger.debug("Extracting run config")
            run_config = self._extract_run_config(settings)
            logger.info(
                "Creating MLflow run '{}' in experiment '{}'",
                run_config.get("run_name") or "<auto>",
                run_config.get("experiment_name") or "DLKit",
            )

            # Execute training within tracking context
            logger.debug("Creating MLflow run")
            pending_registration = None
            enriched_result = None
            with self._tracker.create_run(**run_config) as run_context:
                logger.debug("MLflow run created successfully")

                # Fire on_run_created hook
                if self._hooks and self._hooks.on_run_created:
                    self._hooks.on_run_created(run_context.run_id, tracking_uri)

                # Log extra params from hooks
                if self._hooks and self._hooks.extra_params:
                    run_context.log_params(self._hooks.extra_params(settings))

                # Log metadata and configuration
                self._log_tracking_metadata(run_context, tracking_uri)
                self._log_configuration(components, settings, run_context)

                # Inject MLflow callbacks into trainer
                self._inject_mlflow_callbacks(components, run_context, settings)

                # Execute core training
                result = self._executor.execute(components, settings)

                # Fire on_training_complete hook
                if self._hooks and self._hooks.on_training_complete:
                    self._hooks.on_training_complete(result)

                # Log final summary metrics (delegate to MetricLogger)
                self._metric_logger.log_summary_metrics(result, run_context)

                # Log training artifacts (delegate to ArtifactLogger)
                pending_registration = self._artifact_logger.log_training_artifacts(
                    components, settings, run_context
                )

                # Log user-defined custom artifacts and params (delegate to ArtifactLogger)
                self._artifact_logger.log_user_artifacts(settings, run_context, result)

                # Log extra artifacts from hooks
                if self._hooks and self._hooks.extra_artifacts:
                    for path in self._hooks.extra_artifacts(result):
                        run_context.log_artifact(path)

                # Enrich result with tracking metadata (delegate to ResultEnricher)
                enriched_result = self._result_enricher.enrich_result(
                    result, settings, tracking_uri, run_context=run_context
                )

            self._artifact_logger.finalize_model_registration(pending_registration, run_context)
            if enriched_result is None:
                raise RuntimeError("Tracking result enrichment failed unexpectedly")
            return enriched_result

        except Exception as e:
            logger.error("Tracking failed: {}", e)
            raise_error("Training with tracking failed", e, stage="tracking")

    def _setup_tracking(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> None:
        """Setup tracking configuration.

        Delegates to tracker's configure() if available (LSP-compliant via Protocol).
        No return value — use get_tracking_uri() after __enter__ instead.

        Args:
            settings: Global settings
        """
        accessor = ConfigAccessor(settings)
        mlflow_config = accessor.get_mlflow_config()

        if isinstance(self._tracker, ITrackingSetup):
            root_dir = accessor.get_session_root_dir()
            try:
                self._tracker.configure(mlflow_config, root_dir=root_dir)
            except TypeError as exc:
                if "unexpected keyword argument 'root_dir'" in str(exc):
                    logger.debug("Tracker does not support root_dir parameter, using fallback")
                    self._tracker.configure(mlflow_config)
                else:
                    raise

    def _extract_run_config(
        self, settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig
    ) -> dict:
        """Extract run configuration from settings including merged tags.

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

        settings_tags: dict[str, str] = accessor.get_run_tags() or {}
        hooks_tags: dict[str, str] = (
            self._hooks.extra_tags(settings) if self._hooks and self._hooks.extra_tags else {}
        )
        # hooks win on collision
        merged_tags = {**settings_tags, **hooks_tags}

        return {
            "experiment_name": determine_experiment_name(settings, mlflow_config),
            "run_name": accessor.get_run_name(),
            "nested": self._should_use_nested_runs(),
            "tags": merged_tags or None,
        }

    def _log_tracking_metadata(
        self,
        run_context: IRunContext,
        tracking_uri: str | None,
    ) -> None:
        """Log tracking metadata to run context.

        Logs resolved tracking URI as an MLflow tag.

        Args:
            run_context: Run context for logging
            tracking_uri: Resolved tracking URI if available
        """
        if tracking_uri:
            run_context.set_tag("mlflow_tracking_uri", tracking_uri)

    def _log_configuration(
        self,
        components: RuntimeComponents,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
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
        components: RuntimeComponents,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        run_context: IRunContext,
    ) -> None:
        """Log dataset to MLflow run.

        Delegates to tracker if it supports dataset logging (LSP-compliant via Protocol).

        Args:
            components: Build components containing datamodule
            settings: Global settings
            run_context: Run context for logging
        """
        try:
            if isinstance(self._tracker, IDatasetLogger):
                self._tracker.log_dataset_to_run(components.datamodule, run_context, settings)
        except Exception as e:
            logger.warning("Failed to log dataset: {}", e)

    def _inject_mlflow_callbacks(
        self,
        components: RuntimeComponents,
        run_context: IRunContext,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> None:
        """Inject MLflow callbacks into trainer.

        Injects ``MLflowEpochLogger`` so metrics are logged with epoch numbers
        as the x-axis during training.

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

            from dlkit.engine.adapters.lightning.callbacks import MLflowEpochLogger

            if not hasattr(trainer, "callbacks"):
                trainer.callbacks = []

            trainer.callbacks.append(MLflowEpochLogger(run_context))
            logger.debug("Injected MLflow epoch logger for run '{}'", run_context.run_id)

        except Exception as e:
            logger.warning("Failed to inject MLflow epoch logger: {}", e)

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
            logger.warning("Failed to check for active MLflow run: {}", e)
            return False
