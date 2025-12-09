"""MLflow tracking adapter implementing proper nested run hierarchy.

This adapter implements the IExperimentTracker protocol to provide proper
nested MLflow run structure for Optuna optimization:
- Parent run: Study
- Child runs: Individual trials
- Final child run: Best parameter retrain
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Any

try:  # MLflow is optional when tracking disabled
    import mlflow  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised via indirect tests
    mlflow = None  # type: ignore[assignment]

from dlkit.interfaces.api.domain import WorkflowError
from dlkit.tools.utils.logging_config import get_logger
from dlkit.runtime.workflows.optimization.domain import (
    Study,
    Trial,
    OptimizationResult,
    IExperimentTracker,
    IStudyRunContext,
    ITrialRunContext,
)

logger = get_logger(__name__)


class MLflowTrackingAdapter(IExperimentTracker):
    """MLflow adapter implementing proper nested run hierarchy for optimization.

    This adapter creates the proper Study → Trial → Best Retrain hierarchy
    by delegating to the existing MLflowTracker which handles server lifecycle.

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
        root_dir: Path | str | None = None,
    ):
        """Initialize MLflow tracking adapter.

        Args:
            mlflow_tracker: Existing MLflowTracker instance with server management
            mlflow_settings: MLflow configuration settings for initialization
            session_name: Session name to use as experiment name
        """
        self._tracker = mlflow_tracker
        self._mlflow_settings = mlflow_settings
        self._session_name = session_name
        self._exit_stack: ExitStack | None = None
        self._root_dir = Path(root_dir).resolve() if root_dir is not None else None
        self._explicit_run_name: str | None = None

        if self._mlflow_settings:
            client = getattr(self._mlflow_settings, "client", None)
            if client is not None:
                candidate = getattr(client, "run_name", None)
                if isinstance(candidate, str):
                    candidate = candidate.strip() or None
                if candidate:
                    self._explicit_run_name = candidate

        if self._tracker is None:
            # Import and create the existing MLflowTracker with server management
            try:
                from dlkit.runtime.workflows.strategies.tracking.mlflow_tracker import MLflowTracker

                self._tracker = MLflowTracker(
                    disable_autostart=False,
                    skip_health_checks=False,
                )
            except ImportError as e:
                raise WorkflowError(
                    f"MLflowTracker not available: {e}", {"stage": "tracking_initialization"}
                ) from e

    def __enter__(self):
        """Enter context and initialize MLflow tracker using ExitStack."""
        logger.info(f"MLflowTrackingAdapter.__enter__ called - settings={self._mlflow_settings is not None}, tracker={self._tracker is not None}")

        if self._mlflow_settings and self._tracker:
            try:
                # Create ExitStack for managing nested context managers
                self._exit_stack = ExitStack()
                self._exit_stack.__enter__()

                # Configure and enter tracker context using ExitStack
                logger.debug("Configuring and entering MLflow tracker context")
                self._tracker.setup_mlflow_config(
                    self._mlflow_settings,
                    root_dir=self._root_dir,
                )
                self._tracker = self._exit_stack.enter_context(self._tracker)

                logger.info("MLflow tracking adapter context entered successfully")
            except Exception as e:
                logger.error(f"Failed to initialize MLflow tracker: {e}")
                if self._exit_stack:
                    self._exit_stack.__exit__(None, None, None)
                    self._exit_stack = None
                raise
        else:
            logger.warning(f"Skipping MLflow initialization - settings={self._mlflow_settings is not None}, tracker={self._tracker is not None}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and cleanup MLflow resources via ExitStack."""
        # If MLflow unavailable, nothing to clean up
        if self._exit_stack:
            try:
                logger.debug("Cleaning up MLflow tracker via ExitStack")
                self._exit_stack.__exit__(exc_type, exc_val, exc_tb)
                logger.info("MLflow tracking adapter context exited")
            except Exception as e:
                logger.warning(f"Failed to exit tracker context: {e}")
            finally:
                self._exit_stack = None
        return False

    @contextmanager
    def create_study_run(self, study: Study):
        """Create parent run for optimization study using existing MLflowTracker."""
        self._ensure_mlflow_available("study_run_creation")
        logger.info(
            "Creating MLflow study run",
            study_id=study.study_id,
            study_name=study.study_name,
        )

        experiment_name = self._get_experiment_name()
        run_name = self._get_run_name_from_study(study)

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=run_name,
                nested=False,  # Parent run
            ) as run_context:
                study_context = MLflowStudyRunContext(mlflow, run_context, study)
                yield study_context

        except Exception as e:
            logger.error("Failed to create study run", error=str(e))
            raise WorkflowError(
                f"Study run creation failed: {e}",
                {"stage": "study_run_creation", "study_id": study.study_id},
            ) from e

    @contextmanager
    def create_trial_run(self, trial: Trial, parent_context: IStudyRunContext):
        """Create nested run for individual trial."""
        self._ensure_mlflow_available("trial_run_creation")
        logger.info(
            "Creating MLflow trial run",
            trial_id=trial.trial_id,
            trial_number=trial.trial_number,
        )

        experiment_name = self._get_experiment_name()

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=f"trial_{trial.trial_number}",
                nested=True,  # Nested under study run
            ) as run_context:
                trial_context = MLflowTrialRunContext(mlflow, run_context, trial)
                yield trial_context

        except Exception as e:
            logger.error("Failed to create trial run", error=str(e))
            raise WorkflowError(
                f"Trial run creation failed: {e}",
                {"stage": "trial_run_creation", "trial_id": trial.trial_id},
            ) from e

    @contextmanager
    def create_best_retrain_run(self, study: Study, parent_context: IStudyRunContext):
        """Create nested run for best parameter retraining."""
        self._ensure_mlflow_available("best_retrain_creation")
        best_trial = study.best_trial
        if not best_trial:
            raise WorkflowError(
                "Cannot create best retrain run without best trial",
                {"stage": "best_retrain_creation", "study_id": study.study_id},
            )

        logger.info(
            "Creating MLflow best retrain run",
            study_id=study.study_id,
            best_trial_number=best_trial.trial_number,
        )

        experiment_name = self._get_experiment_name()

        try:
            with self._tracker.create_run(
                experiment_name=experiment_name,
                run_name=f"best_retrain_trial_{best_trial.trial_number}",
                nested=True,  # Nested under study run
            ) as run_context:
                retrain_context = MLflowTrialRunContext(
                    mlflow, run_context, best_trial
                )
                yield retrain_context

        except Exception as e:
            logger.error("Failed to create best retrain run", error=str(e))
            raise WorkflowError(
                f"Best retrain run creation failed: {e}",
                {"stage": "best_retrain_creation", "study_id": study.study_id},
            ) from e

    def _get_experiment_name(self) -> str:
        """Get experiment name.

        Returns:
            Experiment name (passed as session_name from factory)
        """
        return self._session_name or "DLKit"

    def _get_run_name_from_study(self, study: Study) -> str | None:
        """Resolve parent MLflow run name for the study.

        Returns the explicit run name only when configured under
        ``MLFLOW.client.run_name``. Otherwise ``None`` is returned so MLflow can
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


class MLflowStudyRunContext(IStudyRunContext):
    """MLflow context for study-level tracking."""

    def __init__(self, mlflow_module: Any, run_context: Any, study: Study):
        """Initialize study run context.

        Args:
            mlflow_module: MLflow module
            run_context: Existing MLflowRunContext from MLflowTracker
            study: Study domain model
        """
        self._mlflow = mlflow_module
        self._run_context = run_context
        self._study = study

    def log_study_metadata(self, study: Study) -> None:
        """Log study-level metadata."""
        try:
            # Log study parameters using the run context
            self._run_context.log_params({
                "study_name": study.study_name,
                "optimization_direction": study.direction.value,
                "target_trials": study.target_trials,
                "study_id": study.study_id,
            })

            # Log sampler configuration
            if study.sampler_config:
                for key, value in study.sampler_config.items():
                    self._run_context.log_params({f"sampler_{key}": value})

            # Log pruner configuration
            if study.pruner_config:
                for key, value in study.pruner_config.items():
                    self._run_context.log_params({f"pruner_{key}": value})

            # Set study tags using direct MLflow access
            self._mlflow.set_tags({
                "optimization_framework": "optuna",
                "optimization_type": "hyperparameter_optimization",
                "study_id": study.study_id,
            })

            logger.debug("Study metadata logged to MLflow")

        except Exception as e:
            logger.warning("Failed to log study metadata", error=str(e))

    def log_study_summary(self, result: OptimizationResult) -> None:
        """Log final study summary."""
        try:
            # Log study-level metrics using run context
            self._run_context.log_metrics({
                "total_trials": float(result.total_trials),
                "successful_trials": float(result.successful_trials),
                "optimization_duration_seconds": result.total_duration_seconds,
            })

            # Log best results if available
            if result.best_objective_value is not None:
                self._run_context.log_metrics({"best_objective_value": result.best_objective_value})

            if result.best_trial:
                self._run_context.log_metrics({
                    "best_trial_number": float(result.best_trial.trial_number)
                })

                # Log best hyperparameters as parameters
                for key, value in result.best_hyperparameters.items():
                    self._run_context.log_params({f"best_{key}": value})

            logger.debug("Study summary logged to MLflow")

        except Exception as e:
            logger.warning("Failed to log study summary", error=str(e))

    def log_best_trial_settings(self, settings: Any) -> None:
        """Log best trial settings as TOML artifact with special naming.

        Args:
            settings: GeneralSettings object for the best trial
        """
        try:
            from dlkit.tools.io.config import write_config
            from pathlib import Path
            import os
            import tempfile

            # Create temporary TOML file with unset values excluded
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".toml", delete=False, prefix="best_trial_config_"
            ) as temp_file:
                write_config(
                    settings,
                    temp_file.name,
                    exclude_unset=True,
                    exclude_value_entries=True,
                )
                temp_path = Path(temp_file.name)

            try:
                # Log TOML file as artifact with clear naming
                self._run_context.log_artifact(temp_path, artifact_dir="")
                logger.debug("Best trial settings logged as TOML artifact")
            finally:
                # Clean up temp file
                if temp_path.exists():
                    os.unlink(temp_path)

        except Exception as e:
            logger.warning("Failed to log best trial settings", error=str(e))


class MLflowTrialRunContext(ITrialRunContext):
    """MLflow context for trial-level tracking."""

    def __init__(self, mlflow_module: Any, run_context: Any, trial: Trial):
        """Initialize trial run context.

        Args:
            mlflow_module: MLflow module
            run_context: Existing MLflowRunContext from MLflowTracker
            trial: Trial domain model
        """
        self._mlflow = mlflow_module
        self._run_context = run_context
        self._trial = trial

    def log_trial_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Log trial hyperparameters.

        IMPORTANT: Only logs static hyperparameters, NOT trial state or other changing values.
        Hyperparameters are values that are set BEFORE training and don't change during execution.
        """
        try:
            # Log hyperparameters as MLflow parameters using run context
            self._run_context.log_params(hyperparameters)

            # Log trial identifier (static, doesn't change during trial)
            self._run_context.log_params({
                "trial_id": self._trial.trial_id,
                "trial_number": self._trial.trial_number,
            })
            # NOTE: trial_state is NOT logged as a parameter because it changes during execution
            # State information should be logged as tags or tracked separately

            logger.debug(
                "Trial hyperparameters logged to MLflow",
                trial_number=self._trial.trial_number,
            )

        except Exception as e:
            logger.warning("Failed to log trial hyperparameters", error=str(e))

    def log_trial_settings(self, settings: Any) -> None:
        """Log trial settings as TOML artifact.

        Args:
            settings: GeneralSettings object for this trial
        """
        try:
            from dlkit.tools.io.config import write_config
            from pathlib import Path
            import os
            import tempfile

            # Create temporary TOML file with unset values excluded
            with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as temp_file:
                write_config(
                    settings,
                    temp_file.name,
                    exclude_unset=True,
                    exclude_value_entries=True,
                )
                temp_path = Path(temp_file.name)

            try:
                # Log TOML file as artifact
                self._run_context.log_artifact(temp_path, artifact_dir="")
                logger.debug(
                    "Trial settings logged as TOML artifact",
                    trial_number=self._trial.trial_number,
                )
            finally:
                # Clean up temp file
                if temp_path.exists():
                    os.unlink(temp_path)

        except Exception as e:
            logger.warning("Failed to log trial settings", error=str(e))

    def log_model_hyperparameters(self, settings: Any) -> None:
        """Log model hyperparameters from settings.MODEL.

        Args:
            settings: GeneralSettings object with MODEL configuration
        """
        try:
            if not hasattr(settings, "MODEL") or settings.MODEL is None:
                return

            # Use settings.MODEL and dump its hyperparameters
            params = settings.MODEL.model_dump(exclude_none=True)

            # Remove component-specific fields that aren't hyperparameters
            component_fields = {"name", "module_path", "checkpoint", "shape"}
            hparams = {k: v for k, v in params.items() if k not in component_fields}

            # Prefix with "model_" to distinguish from trial hyperparameters
            prefixed_hparams = {f"model_{k}": v for k, v in hparams.items()}

            # Only log if we have hyperparameters to log
            if prefixed_hparams:
                self._run_context.log_params(prefixed_hparams)
                logger.debug(
                    "Model hyperparameters logged to MLflow",
                    trial_number=self._trial.trial_number,
                    hparam_count=len(prefixed_hparams),
                )

        except Exception as e:
            logger.warning("Failed to log model hyperparameters", error=str(e))

    def log_trial_metrics(self, metrics: dict[str, Any]) -> None:
        """Log trial metrics."""
        try:
            # Filter numeric metrics for MLflow
            numeric_metrics = {}
            for key, value in metrics.items():
                try:
                    numeric_metrics[key] = float(value)
                except (ValueError, TypeError):
                    # Log non-numeric as parameters
                    self._run_context.log_params({f"metric_{key}": str(value)})

            if numeric_metrics:
                self._run_context.log_metrics(numeric_metrics)

            # Log trial-specific metrics
            if self._trial.objective_value is not None:
                self._run_context.log_metrics({"objective_value": self._trial.objective_value})

            if self._trial.duration_seconds > 0:
                self._run_context.log_metrics({
                    "trial_duration_seconds": self._trial.duration_seconds
                })

            logger.debug(
                "Trial metrics logged to MLflow",
                trial_number=self._trial.trial_number,
                metrics_count=len(numeric_metrics),
            )

        except Exception as e:
            logger.warning("Failed to log trial metrics", error=str(e))

    def log_trial_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Log trial artifacts."""
        try:
            from pathlib import Path as PathLib

            for key, artifact_path in artifacts.items():
                if hasattr(artifact_path, "exists") and artifact_path.exists():
                    # Log file artifacts using run context
                    self._run_context.log_artifact(PathLib(str(artifact_path)), artifact_dir=key)
                else:
                    # Log as text artifact - write to temp file first
                    temp_path = PathLib("/tmp") / f"{key}.txt"
                    temp_path.write_text(str(artifact_path))
                    self._run_context.log_artifact(temp_path, artifact_dir="")

            logger.debug(
                "Trial artifacts logged to MLflow",
                trial_number=self._trial.trial_number,
                artifacts_count=len(artifacts),
            )

        except Exception as e:
            logger.warning("Failed to log trial artifacts", error=str(e))


class NullTrackingAdapter(IExperimentTracker):
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
        yield NullStudyRunContext()

    @contextmanager
    def create_trial_run(self, trial: Trial, parent_context: IStudyRunContext):
        """Create null trial run context."""
        yield NullTrialRunContext()

    @contextmanager
    def create_best_retrain_run(self, study: Study, parent_context: IStudyRunContext):
        """Create null best retrain run context."""
        yield NullTrialRunContext()


class NullStudyRunContext(IStudyRunContext):
    """Null object implementation for study run context."""

    def log_study_metadata(self, study: Study) -> None:
        """No-op study metadata logging."""
        pass

    def log_study_summary(self, result: OptimizationResult) -> None:
        """No-op study summary logging."""
        pass

    def log_best_trial_settings(self, settings: Any) -> None:
        """No-op best trial settings logging."""
        pass


class NullTrialRunContext(ITrialRunContext):
    """Null object implementation for trial run context."""

    def log_trial_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """No-op hyperparameters logging."""
        pass

    def log_trial_metrics(self, metrics: dict[str, Any]) -> None:
        """No-op metrics logging."""
        pass

    def log_trial_artifacts(self, artifacts: dict[str, Any]) -> None:
        """No-op artifacts logging."""
        pass

    def log_trial_settings(self, settings: Any) -> None:
        """No-op trial settings logging."""
        pass

    def log_model_hyperparameters(self, settings: Any) -> None:
        """No-op model hyperparameters logging."""
        pass
