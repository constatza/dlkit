"""Composable tracking decorator following OCP and SRP.

Coordinates experiment tracking lifecycle by composing focused services.
Each service has a single responsibility following SOLID principles.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from dlkit.common import TrainingResult
from dlkit.common.hooks import LifecycleHooks
from dlkit.engine.artifacts import ArtifactPolicy, NestedRunCapability
from dlkit.engine.tracking.artifact_logger import ArtifactLogger
from dlkit.engine.tracking.config_accessor import ConfigAccessor
from dlkit.engine.tracking.dataset_logger import DatasetLogger
from dlkit.engine.tracking.interfaces import IExperimentTracker, IRunContext
from dlkit.engine.tracking.metric_logger import MetricLogger
from dlkit.engine.tracking.result_enricher import ResultEnricher
from dlkit.engine.tracking.settings_logger import SettingsLogger
from dlkit.engine.training.components import RuntimeComponents
from dlkit.engine.training.interfaces import ITrainingExecutor
from dlkit.infrastructure.config.job_config import JobConfig
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
        settings_logger: SettingsLogger | None = None,
        dataset_logger: DatasetLogger | None = None,
    ):
        """Initialize with base executor, tracker, optional hooks, and optional services.

        Args:
            executor: Core training executor to decorate
            tracker: Experiment tracker implementation
            hooks: Optional functional extension points for lifecycle events
            metric_logger: Metric logging service (optional)
            artifact_logger: Artifact logging service (optional)
            result_enricher: Result enrichment service (optional)
            settings_logger: Settings/model-param serialization service (optional)
            dataset_logger: Dataset lineage logging service (optional)
        """
        self._executor = executor
        self._tracker = tracker
        self._hooks = hooks
        self._metric_logger = metric_logger or MetricLogger()
        self._artifact_logger = artifact_logger or ArtifactLogger(tracker)
        self._result_enricher = result_enricher or ResultEnricher()
        self._settings_logger = settings_logger or SettingsLogger()
        self._dataset_logger = dataset_logger or DatasetLogger()

    def execute(
        self,
        components: RuntimeComponents,
        settings: object,
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
            settings: Global training settings (must be a JobConfig instance)

        Returns:
            TrainingResult enriched with tracking metadata

        Raises:
            WorkflowError: If training or tracking fails
            TypeError: If settings is not a JobConfig instance
        """
        if not isinstance(settings, JobConfig):
            raise TypeError(
                f"TrackingDecorator.execute requires a JobConfig, got {type(settings).__name__}"
            )
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
            tracked_components = self._with_artifact_policy(
                components,
                tracking_enabled=True,
                is_local=is_local,
            )
            if tracking_uri:
                if is_local:
                    logger.info("Using local file-backed MLflow tracking URI: {}", tracking_uri)
                else:
                    logger.info("Using MLflow server tracking URI: {}", tracking_uri)
            logger.debug("Executing training with tracking")
            result = self._execute_with_tracking(
                components,
                tracked_components,
                settings,
                tracking_uri,
            )
            logger.debug("Training completed, exiting tracker context")
            return result

    def _execute_with_tracking(
        self,
        execution_components: RuntimeComponents,
        tracked_components: RuntimeComponents,
        settings: JobConfig,
        tracking_uri: str | None = None,
    ) -> TrainingResult:
        """Execute training with tracking - separated for SRP compliance.

        Args:
            execution_components: Pre-built training components passed to the executor
            tracked_components: Tracking-enriched component view used for artifact policy
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
            enriched_result = None
            with self._tracker.create_run(**run_config) as run_context:
                logger.debug("MLflow run created successfully")

                # Fire on_run_created hook
                if self._hooks and self._hooks.on_run_created:
                    self._hooks.on_run_created(run_context.run_id, tracking_uri)

                # Log metadata and configuration
                self._log_tracking_metadata(run_context, tracking_uri)
                self._log_configuration(tracked_components, settings, run_context)

                # Inject MLflow callbacks into trainer
                self._inject_mlflow_callbacks(tracked_components, run_context, settings)

                # Execute core training
                result = self._executor.execute(execution_components, settings)

                # Fire on_training_complete hook
                if self._hooks and self._hooks.on_training_complete:
                    self._hooks.on_training_complete(result)

                # Log extra params and tags from hooks (post-training, receive result)
                if self._hooks and self._hooks.extra_params:
                    run_context.log_params(self._hooks.extra_params(result))
                if self._hooks and self._hooks.extra_tags:
                    for key, value in self._hooks.extra_tags(result).items():
                        run_context.set_tag(key, value)

                # Log final summary metrics (delegate to MetricLogger)
                self._metric_logger.log_summary_metrics(result, run_context)

                # Log training artifacts (delegate to ArtifactLogger)
                self._artifact_logger.log_training_artifacts(
                    tracked_components, settings, run_context
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
            if enriched_result is None:
                raise RuntimeError("Tracking result enrichment failed unexpectedly")
            return enriched_result

        except Exception as e:
            logger.error("Tracking failed: {}", e)
            raise_error("Training with tracking failed", e, stage="tracking")

    def _setup_tracking(
        self,
        settings: JobConfig,
    ) -> None:
        """Configure the tracker before entering its context.

        Calls ``configure(tracking)`` on the tracker using the backend connection
        settings from the job config.

        Args:
            settings: Global settings (JobConfig instance)
        """
        self._tracker.configure(settings.tracking)

    def _extract_run_config(self, settings: JobConfig) -> dict:
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

        return {
            "experiment_name": determine_experiment_name(settings, mlflow_config),
            "run_name": accessor.get_run_name(),
            "nested": self._should_use_nested_runs(),
            "tags": settings_tags or None,
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
        settings: JobConfig,
        run_context: IRunContext,
    ) -> None:
        """Log configuration, dataset lineage, and model parameters.

        Delegates each concern to a dedicated injected service.

        Args:
            components: Build components
            settings: Global settings
            run_context: Run context for logging
        """
        self._settings_logger.log_settings(settings, run_context)
        self._log_dataset_to_run(components, settings, run_context)
        self._settings_logger.log_model_parameters(components.model, run_context, settings)

    def _log_dataset_to_run(
        self,
        components: RuntimeComponents,
        settings: JobConfig,
        run_context: IRunContext,
    ) -> None:
        """Log dataset lineage via the injected DatasetLogger.

        Args:
            components: Build components containing datamodule
            settings: Global settings
            run_context: Run context for logging
        """
        try:
            self._dataset_logger.log_dataset_to_run(components.datamodule, run_context, settings)
        except Exception as e:
            logger.warning("Failed to log dataset: {}", e)

    def _inject_mlflow_callbacks(
        self,
        components: RuntimeComponents,
        run_context: IRunContext,
        settings: JobConfig,
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

            from dlkit.engine.adapters.lightning.callbacks import (
                CheckpointDirRouter,
                MLflowEpochLogger,
            )

            if not hasattr(trainer, "callbacks"):
                trainer.callbacks = []

            trainer.callbacks.append(MLflowEpochLogger(run_context))
            logger.debug("Injected MLflow epoch logger for run '{}'", run_context.run_id)
            checkpoint_dir = self._resolve_checkpoint_dir(components)
            if checkpoint_dir is not None:
                trainer.callbacks.append(CheckpointDirRouter(checkpoint_dir))
                logger.debug("Injected checkpoint dir router for '{}'", checkpoint_dir)

        except Exception as e:
            logger.warning("Failed to inject MLflow epoch logger: {}", e)

    def _should_use_nested_runs(self) -> bool:
        """Determine if nested runs should be used based on existing MLflow run context.

        Returns:
            True if there's already an active MLflow run (nested run required)
        """
        if isinstance(self._tracker, NestedRunCapability):
            return self._tracker.has_active_parent_run()
        return False

    def _with_artifact_policy(
        self,
        components: RuntimeComponents,
        *,
        tracking_enabled: bool,
        is_local: bool,
    ) -> RuntimeComponents:
        policy = ArtifactPolicy(
            tracking_backend="mlflow" if tracking_enabled else "none",
            checkpoint_persistence="tracked" if tracking_enabled else "framework_local",
            config_persistence="tracked" if tracking_enabled else "none",
            local_root_dir=self._resolve_local_root_dir(components),
            remove_uploaded_files=tracking_enabled and not is_local,
        )
        return replace(components, artifacts=components.artifacts.with_policy(policy))

    @staticmethod
    def _resolve_local_root_dir(components: RuntimeComponents) -> Path | None:
        trainer = components.trainer
        if trainer is None:
            return None
        root_dir = getattr(trainer, "default_root_dir", None)
        if root_dir is None or not isinstance(root_dir, str | Path):
            return None
        return Path(root_dir)

    def _resolve_checkpoint_dir(self, components: RuntimeComponents) -> Path | None:
        local_root_dir = components.artifacts.policy.local_root_dir
        if local_root_dir is None:
            return None
        return local_root_dir / "checkpoints"
