"""MLflow tracking adapter implementing proper nested run hierarchy.

This adapter implements the IStudyTracker protocol to provide proper
nested MLflow run structure for Optuna optimization:
- Parent run: Study
- Child runs: Individual trials
- Final child run: Best parameter retrain
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any

import mlflow

from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks, RunCreatedEvent, RunKind
from dlkit.engine.tracking.artifact_logger import TAG_MODEL_CLASS
from dlkit.engine.tracking.best_effort import best_effort
from dlkit.engine.tracking.config_accessor import ConfigAccessor
from dlkit.engine.tracking.interfaces import NullRunContext
from dlkit.engine.tracking.metric_logger import MetricLogger, split_stage_filtered_metrics
from dlkit.engine.workflows.optimization.value_objects import (
    IStudyTracker,
    OptimizationResult,
    Study,
    Trial,
)
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


class MLflowTrackingAdapter(IStudyTracker):
    """MLflow adapter implementing proper nested run hierarchy for optimization.

    This adapter creates the proper Study → Trial → Best Retrain hierarchy
    by delegating to the existing MLflowTracker client lifecycle.

    Usage:
        with adapter:
            with adapter.create_study_run(study) as study_context:
                ...
    """

    def __init__(
        self,
        mlflow_tracker: Any = None,
        mlflow_settings: Any = None,
        session_name: str | None = None,
        hooks: LifecycleHooks | None = None,
    ):
        """Initialize MLflow tracking adapter.

        Args:
            mlflow_tracker: Existing MLflowTracker instance
            mlflow_settings: Lowercase tracking/experiment config used for initialization
            session_name: Fallback experiment name when the job config does not provide one
            hooks: Optional lifecycle hooks fired around created runs
        """
        self._tracker = mlflow_tracker
        self._mlflow_settings = mlflow_settings
        self._session_name = session_name
        self._hooks = hooks
        self._exit_stack: ExitStack | None = None
        self._explicit_run_name: str | None = None

        if self._mlflow_settings:
            candidate = getattr(self._mlflow_settings, "run_name", None)
            if isinstance(candidate, str):
                candidate = candidate.strip() or None
            if candidate:
                self._explicit_run_name = candidate

        if self._tracker is None:
            # Import and create the existing MLflowTracker
            try:
                from dlkit.engine.tracking.mlflow_tracker import MLflowTracker

                self._tracker = MLflowTracker(disable_autostart=False)
            except ImportError as e:
                raise WorkflowError(
                    f"MLflowTracker not available: {e}", {"stage": "tracking_initialization"}
                ) from e

        # Configure tracker if settings provided
        if self._mlflow_settings and self._tracker:
            self._tracker.configure(self._mlflow_settings)

    def __enter__(self):
        """Enter context and initialize MLflow tracker using ExitStack."""
        logger.debug(
            "MLflow tracking adapter entering (settings={}, tracker={})",
            self._mlflow_settings is not None,
            self._tracker is not None,
        )

        if self._mlflow_settings and self._tracker:
            try:
                # Create ExitStack for managing nested context managers
                self._exit_stack = ExitStack()
                self._exit_stack.__enter__()

                logger.debug("Entering MLflow tracker context")
                self._tracker = self._exit_stack.enter_context(self._tracker)

                logger.debug("MLflow tracking adapter context entered successfully")
            except Exception as e:
                logger.error("Failed to initialize MLflow tracker: {}", e)
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                raise
        else:
            logger.debug(
                "Skipping MLflow initialization (settings={}, tracker={})",
                self._mlflow_settings is not None,
                self._tracker is not None,
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup MLflow resources via ExitStack."""
        # If MLflow unavailable, nothing to clean up
        if self._exit_stack:
            try:
                logger.debug("Cleaning up MLflow tracker via ExitStack")
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                logger.debug("MLflow tracking adapter context exited")
            except Exception as e:
                logger.warning("Failed to exit tracker context: {}", e)
            finally:
                self._exit_stack = None
        return False

    def _notify_run_created(self, run_context: Any, *, kind: RunKind, is_outermost: bool) -> None:
        """Fire ``on_run_created`` hook for a newly created run, if configured."""
        if not self._hooks or not self._hooks.on_run_created:
            return
        tracking_uri = (
            self._tracker.get_tracking_uri() if hasattr(self._tracker, "get_tracking_uri") else None
        )
        self._hooks.on_run_created(
            RunCreatedEvent(
                run_id=run_context.run_id,
                tracking_uri=tracking_uri,
                kind=kind,
                is_outermost=is_outermost,
            )
        )

    @contextmanager
    def create_study_run(self, study: Study):
        """Create parent run for optimization study using existing MLflowTracker."""
        self._ensure_mlflow_available("study_run_creation")
        logger.info("Creating MLflow study run '{}' ({})", study.study_name, study.study_id)

        experiment_name = self._get_experiment_name()
        run_name = self._get_run_name_from_study(study)

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=run_name,
                nested=False,  # Parent run
            ) as run_context:
                self._notify_run_created(run_context, kind="study", is_outermost=True)
                yield run_context

        except Exception as e:
            logger.error("Failed to create study run '{}': {}", study.study_name, e)
            raise WorkflowError(
                f"Study run creation failed: {e}",
                {"stage": "study_run_creation", "study_id": study.study_id},
            ) from e

    @contextmanager
    def create_trial_run(self, trial: Trial, parent_context: Any):
        """Create nested run for individual trial."""
        self._ensure_mlflow_available("trial_run_creation")
        logger.info("Creating MLflow trial run {}", trial.trial_number)

        experiment_name = self._get_experiment_name()

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=f"trial_{trial.trial_number}",
                nested=True,  # Nested under study run
            ) as run_context:
                self._notify_run_created(run_context, kind="trial", is_outermost=False)
                yield run_context

        except Exception as e:
            logger.error("Failed to create trial run {}: {}", trial.trial_number, e)
            raise WorkflowError(
                f"Trial run creation failed: {e}",
                {"stage": "trial_run_creation", "trial_id": trial.trial_id},
            ) from e

    @contextmanager
    def create_best_retrain_run(self, study: Study, parent_context: Any):
        """Create nested run for best parameter retraining."""
        self._ensure_mlflow_available("best_retrain_creation")
        best_trial = study.best_trial
        if not best_trial:
            raise WorkflowError(
                "Cannot create best retrain run without best trial",
                {"stage": "best_retrain_creation", "study_id": study.study_id},
            )

        logger.info("Creating MLflow best-retrain run for trial {}", best_trial.trial_number)

        experiment_name = self._get_experiment_name()

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=f"best_retrain_trial_{best_trial.trial_number}",
                nested=True,  # Nested under study run
            ) as run_context:
                self._notify_run_created(run_context, kind="best_retrain", is_outermost=False)
                yield run_context

        except Exception as e:
            logger.error("Failed to create best retrain run for study '{}': {}", study.study_id, e)
            raise WorkflowError(
                f"Best retrain run creation failed: {e}",
                {"stage": "best_retrain_creation", "study_id": study.study_id},
            ) from e

    def execution_tracker(self) -> Any:
        """Return the underlying ``MLflowTracker`` for use by callers such as
        the trial executor's best-retrain leg, which needs a raw tracker
        instance to pass to ``TrackingDecorator``.

        Returns:
            The wrapped ``MLflowTracker`` instance.
        """
        return self._tracker

    def _get_experiment_name(self) -> str:
        """Get experiment name.

        Returns:
            Experiment name (passed as session_name from factory)
        """
        return self._session_name or "DLKit"

    def _get_run_name_from_study(self, study: Study) -> str | None:
        """Resolve parent MLflow run name for the study.

        Returns the explicit run name only when configured under
        ``experiment.run_name``. Otherwise ``None`` is returned so MLflow can
        generate a random run name instead of mirroring the experiment name.

        Args:
            study: Study domain model (required by interface)

        Returns:
            Configured run name or ``None`` for MLflow auto-naming.
        """
        if self._explicit_run_name:
            return self._explicit_run_name

        candidate = getattr(study, "study_name", None)
        if isinstance(candidate, str):
            candidate = candidate.strip() or None
        return candidate

    def _ensure_mlflow_available(self, stage: str) -> None:
        """Raise an informative error when MLflow is not installed."""
        if mlflow is None:
            raise WorkflowError(
                "MLflow is required for tracking but is not installed",
                {"stage": stage, "dependency": "mlflow"},
            )


class NullTrackingAdapter(IStudyTracker):
    """Null object implementation for when tracking is disabled.

    This eliminates conditional logic throughout the codebase by providing
    safe no-op implementations of all tracking operations. Implements
    AbstractContextManager protocol with no-op __enter__/__exit__ to provide
    uniform interface with MLflowTrackingAdapter.
    """

    def __enter__(self):
        """No-op context entry for null tracker."""
        logger.debug("NullTrackingAdapter context entered (no-op)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """No-op context exit for null tracker."""
        logger.debug("NullTrackingAdapter context exited (no-op)")
        return False

    @contextmanager
    def create_study_run(self, study: Study):
        """Create null study run context."""
        yield NullRunContext()

    @contextmanager
    def create_trial_run(self, trial: Trial, parent_context: Any):
        """Create null trial run context."""
        yield NullRunContext()

    @contextmanager
    def create_best_retrain_run(self, study: Study, parent_context: Any):
        """Create null best retrain run context."""
        yield NullRunContext()


@best_effort("log study metadata")
def log_study_metadata(study: Study, run_context: Any) -> None:
    """Log study-level metadata.

    Args:
        study: Study domain model.
        run_context: Active run context to log against.
    """
    # Log study parameters using the run context
    run_context.log_params(
        {
            "study_name": study.study_name,
            "optimization_direction": study.direction.value,
            "target_trials": study.target_trials,
            "study_id": study.study_id,
        }
    )

    # Log sampler configuration
    if study.sampler_config:
        for key, value in study.sampler_config.items():
            run_context.log_params({f"sampler_{key}": value})

    # Log pruner configuration
    if study.pruner_config:
        for key, value in study.pruner_config.items():
            run_context.log_params({f"pruner_{key}": value})

    # Set study tags using the run context
    for key, value in {
        "optimization_framework": "optuna",
        "optimization_type": "hyperparameter_optimization",
        "study_id": study.study_id,
    }.items():
        run_context.set_tag(key, value)

    logger.debug("Study metadata logged to MLflow")


@best_effort("log study summary")
def log_study_summary(result: OptimizationResult, run_context: Any) -> None:
    """Log final study summary.

    Args:
        result: Optimization result.
        run_context: Active run context to log against.
    """
    # Log study-level metrics using run context
    run_context.log_metrics(
        {
            "total_trials": float(result.total_trials),
            "successful_trials": float(result.successful_trials),
            "optimization_duration_seconds": result.total_duration_seconds,
        }
    )

    # Log best results if available
    if result.best_objective_value is not None:
        run_context.log_metrics({"best_objective_value": result.best_objective_value})

    if result.best_trial:
        run_context.log_metrics({"best_trial_number": float(result.best_trial.trial_number)})

        # Log best hyperparameters as parameters
        for key, value in result.best_hyperparameters.items():
            run_context.log_params({f"best_{key}": value})

    logger.debug("Study summary logged to MLflow")


@best_effort("log best trial settings")
def log_best_trial_settings(settings: Any, run_context: Any) -> None:
    """Log best trial settings as TOML artifact with special naming.

    Args:
        settings: `SearchJobConfig`-derived settings object for the best trial.
        run_context: Active run context to log against.
    """
    from dlkit.infrastructure.io import serialize_config_to_string

    toml_content = serialize_config_to_string(
        settings,
        exclude_unset=True,
        exclude_value_entries=True,
    )
    run_context.log_artifact_content(toml_content, "best_trial_config.toml")
    logger.debug("Best trial settings logged as TOML artifact")


@best_effort("log best trial result")
def log_best_trial_result(training_result: Any, run_context: Any) -> None:
    """Log the best trial's full training result onto the outer study run.

    Without this, the study (parent) run only ever carries aggregate stats
    (``log_study_summary``) and the best trial's config
    (``log_best_trial_settings``) — the actual metrics/artifacts/checkpoint
    only exist on the best-retrain's own nested child run. This copies them
    onto the parent too, so the outer run alone looks like a regular
    training result (unfiltered: no ``MLflowEpochLogger`` runs against the
    study run, so nothing else would ever log these metrics there).

    Args:
        training_result: The best trial's ``TrainingResult`` (already
            produced by the best-retrain leg).
        run_context: Study (outer/parent) run context to log against.
    """
    MetricLogger().log_all_metrics(training_result, run_context)
    log_trial_artifacts(training_result.artifacts or {}, run_context)
    logger.debug("Best trial result logged onto the study run")


@best_effort("log trial settings artifact")
def _log_trial_settings_artifact(settings: Any, run_context: Any) -> None:
    """Log trial settings as a TOML artifact.

    Args:
        settings: `SearchJobConfig`-derived settings object for this trial.
        run_context: Active run context to log against.
    """
    from dlkit.infrastructure.io import serialize_config_to_string

    toml_content = serialize_config_to_string(
        settings,
        exclude_unset=True,
        exclude_value_entries=True,
    )
    run_context.log_artifact_content(toml_content, "trial_config.toml")
    logger.debug("Trial settings logged as TOML artifact")


@best_effort("tag trial model class")
def _log_trial_model_tag(settings: Any, run_context: Any) -> None:
    """Tag the run with the model class used for this trial.

    Args:
        settings: `SearchJobConfig`-derived settings object for this trial.
        run_context: Active run context to log against.
    """
    run_context.set_tag(TAG_MODEL_CLASS, ConfigAccessor(settings).get_model_name())


def log_trial_settings(settings: Any, run_context: Any) -> None:
    """Log trial settings as TOML artifact and tag the run with the model class.

    Trial runs never go through the main-training artifact-logging path, so
    without this tag there is no way to tell which model class a trial used
    once its run name is uninformative.

    Args:
        settings: `SearchJobConfig`-derived settings object for this trial.
        run_context: Active run context to log against.
    """
    _log_trial_settings_artifact(settings, run_context)
    _log_trial_model_tag(settings, run_context)


@best_effort("log model hyperparameters")
def log_model_hyperparameters(settings: Any, run_context: Any) -> None:
    """Log model hyperparameters from settings.model.

    Args:
        settings: JobConfig object with model configuration.
        run_context: Active run context to log against.
    """
    model_cfg = getattr(settings, "model", None)
    if model_cfg is None:
        return

    params = model_cfg.model_dump(exclude_none=True)

    # Remove component-specific fields that aren't hyperparameters
    component_fields = {"name", "module_path", "checkpoint", "shape"}
    hparams = {k: v for k, v in params.items() if k not in component_fields}

    # Prefix with "model_" to distinguish from trial hyperparameters
    prefixed_hparams = {f"model_{k}": v for k, v in hparams.items()}

    if prefixed_hparams:
        run_context.log_params(prefixed_hparams)
        logger.debug("Model hyperparameters logged to MLflow")


@best_effort("log trial hyperparameters")
def log_trial_hyperparameters(
    hyperparameters: dict[str, Any], trial: Trial, run_context: Any
) -> None:
    """Log trial hyperparameters.

    IMPORTANT: Only logs static hyperparameters, NOT trial state or other changing values.
    Hyperparameters are values that are set BEFORE training and don't change during execution.

    Keys are logged with their top-level section prefix (``model.``,
    ``training.``, ...) stripped, since that segment only exists to route the
    value into the right JobConfig field at patch time and is noise in the
    MLflow UI. The rest of the path is kept so sibling leaves under different
    sections don't collide (``training.optimizer.lr`` -> ``optimizer.lr``).

    Args:
        hyperparameters: Trial hyperparameters.
        trial: Trial domain model.
        run_context: Active run context to log against.
    """
    display_hyperparameters = {
        key.split(".", 1)[1] if "." in key else key: value for key, value in hyperparameters.items()
    }
    # Log hyperparameters as MLflow parameters using run context
    run_context.log_params(display_hyperparameters)

    # Log trial identifier (static, doesn't change during trial)
    run_context.log_params(
        {
            "trial_id": trial.trial_id,
            "trial_number": trial.trial_number,
        }
    )
    # NOTE: trial_state is NOT logged as a parameter because it changes during execution
    # State information should be logged as tags or tracked separately

    logger.debug("Trial {} hyperparameters logged to MLflow", trial.trial_number)


@best_effort("log trial metrics")
def log_trial_metrics(metrics: dict[str, Any], run_context: Any) -> None:
    """Log trial metrics, excluding train/val metrics already logged per-epoch.

    ``MLflowEpochLogger`` (injected into the trial's trainer) already logs
    train/val metrics with epoch steps. Test metrics are a single scalar over
    the whole test set, so they are logged here, once, with no step.

    Args:
        metrics: Trial metrics.
        run_context: Active run context to log against.
    """
    numeric_metrics, fallback_metrics = split_stage_filtered_metrics(metrics)

    if fallback_metrics:
        run_context.log_params(fallback_metrics)

    if numeric_metrics:
        run_context.log_metrics(numeric_metrics)

    logger.debug("Trial metrics logged to MLflow")


@best_effort("log trial outcome")
def log_trial_outcome(trial: Trial, run_context: Any) -> None:
    """Log trial-specific outcome metrics (objective value and duration).

    Args:
        trial: Trial domain model with final state.
        run_context: Active run context to log against.
    """
    if trial.objective_value is not None:
        run_context.log_metrics({"objective_value": trial.objective_value})

    if trial.duration_seconds > 0:
        run_context.log_metrics({"trial_duration_seconds": trial.duration_seconds})

    logger.debug("Trial {} outcome logged to MLflow", trial.trial_number)


@best_effort("log trial artifacts")
def log_trial_artifacts(artifacts: dict[str, Any], run_context: Any) -> None:
    """Log trial artifacts.

    Args:
        artifacts: Trial artifacts.
        run_context: Active run context to log against.
    """
    for key, artifact_path in artifacts.items():
        if hasattr(artifact_path, "exists") and artifact_path.exists():
            # Log file artifacts using run context
            run_context.log_artifact(artifact_path, artifact_dir=key)
        else:
            run_context.log_artifact_content(str(artifact_path), f"{key}.txt")

    logger.debug("Trial artifacts logged to MLflow")
