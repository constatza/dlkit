"""Application services for optimization orchestration following SOLID principles.

These services coordinate between domain models, repositories, and infrastructure
adapters to execute optimization workflows. Each service has a single responsibility
and depends on abstractions rather than concrete implementations.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, cast

from dlkit.common import TrainingResult
from dlkit.common.errors import WorkflowError
from dlkit.common.hooks import LifecycleHooks
from dlkit.common.metric_stages import DEFAULT_VAL_LOSS_METRIC
from dlkit.engine.tracking.lightweight_execution import (
    execute_lightweight,
    fire_post_training_hooks,
)
from dlkit.engine.workflows.factories.build_factory import BuildFactory
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.utils.logging_config import get_logger

from ._trial_helpers import complete_trial, fail_trial, prune_trial
from .infrastructure.tracking import (
    log_best_trial_result,
    log_best_trial_settings,
    log_study_metadata,
    log_study_summary,
    log_trial_artifacts,
    log_trial_hyperparameters,
    log_trial_metrics,
    log_trial_outcome,
    log_trial_settings,
)
from .value_objects import (
    IConfigurationPersistence,
    IExperimentTracker,
    IHyperparameterApplicator,
    IOptimizationBackendSession,
    IStudyRepository,
    OptimizationDirection,
    OptimizationResult,
    Study,
    Trial,
    TrialFailedException,
    TrialPrunedException,
)

logger = get_logger(__name__)


class StudyManager:
    """Service responsible for Study lifecycle management.

    This service handles creating, managing, and persisting optimization studies
    following the Single Responsibility Principle.
    """

    def __init__(self, repository: IStudyRepository):
        """Initialize study manager with repository.

        Args:
            repository: Study repository implementation
        """
        self._repository = repository

    def create_study(
        self,
        study_name: str,
        direction: OptimizationDirection,
        target_trials: int,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> Study:
        """Create a new optimization study.

        Args:
            study_name: Name of the study
            direction: Optimization direction
            target_trials: Number of trials to run
            sampler_config: Sampler configuration
            pruner_config: Pruner configuration
            storage_config: Storage configuration

        Returns:
            Created study
        """
        logger.info(
            "Creating optimization study '{}' (direction={}, trials={})",
            study_name,
            direction.value,
            target_trials,
        )

        try:
            study = self._repository.create_study(
                study_name=study_name,
                direction=direction,
                target_trials=target_trials,
                sampler_config=sampler_config,
                pruner_config=pruner_config,
                storage_config=storage_config,
            )

            logger.info(
                "Created optimization study '{}' ({})",
                study.study_name,
                study.study_id,
            )

            return study

        except Exception as e:
            logger.error("Failed to create study '{}': {}", study_name, e)
            raise WorkflowError(
                f"Study creation failed: {e}", {"stage": "study_creation", "study_name": study_name}
            ) from e

    def get_study(self, study_id: str) -> Study | None:
        """Get study by ID.

        Args:
            study_id: Study identifier

        Returns:
            Study if found, None otherwise
        """
        return self._repository.get_study(study_id)

    def save_study(self, study: Study) -> None:
        """Save study to repository.

        Args:
            study: Study to save
        """
        self._repository.save_study(study)

    def complete_study(self, study_id: str) -> None:
        """Mark study as completed.

        Args:
            study_id: Study identifier
        """
        study = self._repository.get_study(study_id)
        if not study:
            raise WorkflowError(
                f"Study not found: {study_id}", {"stage": "study_completion", "study_id": study_id}
            )

        study = study.complete_study()
        self._repository.save_study(study)

        logger.info(
            "Study '{}' completed with {}/{} successful trials",
            study_id,
            len(study.successful_trials),
            len(study.trials),
        )


class TrialExecutor:
    """Service responsible for executing individual optimization trials.

    This service handles the execution of single trials, including hyperparameter
    sampling, training execution, and result collection.
    """

    def __init__(
        self,
        build_factory: BuildFactory,
        hyperparameter_applicator: IHyperparameterApplicator | None = None,
    ):
        """Initialize trial executor.

        Args:
            build_factory: Factory for building training components
            hyperparameter_applicator: Optional hyperparameter applicator (default: ModelSettingsApplicator)
        """
        self._build_factory = build_factory
        if hyperparameter_applicator is None:
            from dlkit.engine.workflows.optimization.infrastructure.applicators import (
                ModelSettingsApplicator,
            )

            hyperparameter_applicator = ModelSettingsApplicator()
        self._hyperparameter_applicator = hyperparameter_applicator

    def execute_trial(
        self,
        trial: Trial,
        base_settings: SearchJobConfig,
        hyperparameters: dict[str, Any],
        trial_context: Any | None = None,
    ) -> TrainingResult:
        """Execute a single optimization trial.

        Args:
            trial: Trial domain model
            base_settings: Base configuration settings
            hyperparameters: Hyperparameters for this trial
            trial_context: Optional trial run context for metric logging

        Returns:
            Training result from trial execution

        Raises:
            TrialPrunedException: If trial should be pruned
            TrialFailedException: If trial execution fails
        """
        logger.info("Starting optimization trial {}", trial.trial_number)

        try:
            # Apply hyperparameters to base settings
            trial_settings = self.apply_hyperparameters(base_settings, hyperparameters)

            # Build components for this trial
            components = self._build_factory.build_components(trial_settings)

            # Execute training with lightweight tracking (no checkpoints, epoch metrics only).
            # Pruning integrates through the optimization backend/trial context path.
            training_result = execute_lightweight(components, trial_settings, trial_context)

            logger.info("Completed optimization trial {}", trial.trial_number)

            return training_result

        except TrialPrunedException:
            logger.info("Pruned optimization trial {}", trial.trial_number)
            raise

        except Exception as e:
            logger.error("Optimization trial {} failed: {}", trial.trial_number, e)
            raise TrialFailedException(f"Trial execution failed: {e}") from e

    def execute_best_retrain(
        self,
        base_settings: SearchJobConfig,
        hyperparameters: dict[str, Any],
        *,
        run_context: Any,
        tracker: Any,
        tracking_uri: str | None,
        hooks: LifecycleHooks | None,
    ) -> tuple[SearchJobConfig, TrainingResult]:
        """Execute the best-trial retrain with full tracking parity.

        Unlike :meth:`execute_trial`, which uses the lightweight execution path
        (no checkpoints, epoch metrics only) for high-volume trial throughput,
        this retrains the best trial through :class:`TrackingDecorator`, giving
        it the same checkpoint/artifact/plot logging as a plain ``train()`` call.

        Args:
            base_settings: Base configuration settings.
            hyperparameters: Best trial's hyperparameters to apply.
            run_context: Already-open run context to log against.
            tracker: Underlying execution-side tracker matching ``run_context``.
            tracking_uri: Resolved tracking URI if available.
            hooks: Optional lifecycle hooks fired around training completion.

        Returns:
            Tuple of the resolved trial settings and the enriched training result.
        """
        from dlkit.engine.tracking.tracking_decorator import TrackingDecorator
        from dlkit.engine.training.vanilla_executor import VanillaExecutor

        trial_settings = self.apply_hyperparameters(base_settings, hyperparameters)
        components = self._build_factory.build_components(trial_settings)

        decorator = TrackingDecorator(VanillaExecutor(), tracker, hooks=hooks)
        result = decorator.execute_within_run(
            components,
            trial_settings,
            run_context=run_context,
            tracking_uri=tracking_uri,
        )
        return trial_settings, result

    def apply_hyperparameters(
        self,
        base_settings: SearchJobConfig,
        hyperparameters: dict[str, Any],
    ) -> SearchJobConfig:
        """Apply hyperparameters to base settings.

        Args:
            base_settings: Base configuration
            hyperparameters: Hyperparameters to apply

        Returns:
            Settings with hyperparameters applied

        Raises:
            WorkflowError: If hyperparameter application fails
        """
        try:
            return cast(
                SearchJobConfig,
                self._hyperparameter_applicator.apply(base_settings, hyperparameters),
            )
        except Exception as e:
            raise WorkflowError(
                f"Failed to apply hyperparameters: {e}",
                {"stage": "hyperparameter_application"},
            ) from e

    def extract_objective_value(
        self, training_result: TrainingResult, objective: str = DEFAULT_VAL_LOSS_METRIC
    ) -> float:
        """Extract objective value from training result.

        Args:
            training_result: Result from training
            objective: Metric key to extract (e.g. "val/loss", "train/loss"),
                normally resolved from ``SearchSettings.objective``.

        Returns:
            Objective value for optimization.

        Raises:
            WorkflowError: If the objective metric key is missing from
                ``training_result.metrics``, or its value can't be cast to
                ``float``. A silent fallback would poison the whole study's
                optimization direction with no signal anything went wrong.
        """
        if objective not in training_result.metrics:
            raise WorkflowError(
                f"Objective metric {objective!r} not found in training result metrics "
                f"({sorted(training_result.metrics)}). Check trial_settings.search.objective "
                "for a typo, or that the metric is actually logged for this dataset.",
                {"stage": "objective_extraction", "objective": objective},
            )

        try:
            return float(training_result.metrics[objective])
        except (ValueError, TypeError) as e:
            raise WorkflowError(
                f"Objective metric {objective!r} has a non-numeric value "
                f"({training_result.metrics[objective]!r})",
                {"stage": "objective_extraction", "objective": objective},
            ) from e


class OptimizationOrchestrator:
    """Main orchestrator service for optimization workflows.

    This service coordinates all aspects of optimization execution including:
    - Study management
    - Trial execution
    - Experiment tracking
    - Configuration persistence

    It follows the Single Responsibility Principle by focusing solely on
    orchestration while delegating specific tasks to specialized services.
    """

    def __init__(
        self,
        study_manager: StudyManager,
        trial_executor: TrialExecutor,
        optimization_backend_session: IOptimizationBackendSession,
        experiment_tracker: IExperimentTracker | None = None,
        config_persister: IConfigurationPersistence | None = None,
        hooks: LifecycleHooks | None = None,
    ):
        """Initialize optimization orchestrator.

        Args:
            study_manager: Study lifecycle management service
            trial_executor: Trial execution service
            optimization_backend_session: Runtime optimization backend session
            experiment_tracker: Optional experiment tracking
            config_persister: Optional configuration persistence
            hooks: Optional lifecycle hooks fired around trial/retrain training
        """
        self._study_manager = study_manager
        self._trial_executor = trial_executor
        self._optimization_backend_session = optimization_backend_session
        self._experiment_tracker = experiment_tracker
        self._config_persister = config_persister
        self._hooks = hooks

    def execute_optimization(
        self,
        study_name: str,
        base_settings: SearchJobConfig,
        n_trials: int,
        direction: OptimizationDirection,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Execute complete optimization workflow.

        The runtime edge owns the experiment tracker context lifecycle. This
        service assumes any tracker dependency has already been entered.

        Args:
            study_name: Name of the optimization study
            base_settings: Base configuration settings
            n_trials: Number of trials to execute
            direction: Optimization direction
            sampler_config: Sampler configuration
            pruner_config: Pruner configuration
            storage_config: Storage configuration

        Returns:
            Complete optimization result

        Raises:
            WorkflowError: If optimization fails
        """
        logger.info(
            "Starting optimization workflow '{}' (direction={}, trials={})",
            study_name,
            direction.value,
            n_trials,
        )

        try:
            tracker = self._experiment_tracker
            study_kwargs = {
                "study_name": study_name,
                "direction": direction,
                "target_trials": n_trials,
                "sampler_config": sampler_config,
                "pruner_config": pruner_config,
                "storage_config": storage_config,
            }

            with self._optimization_backend_session:
                study = self._study_manager.create_study(**study_kwargs)
                if tracker is not None:
                    return self._execute_with_tracking(study, base_settings)
                return self._execute_without_tracking(study, base_settings)

        except Exception as e:
            logger.error("Optimization workflow '{}' failed: {}", study_name, e)
            raise WorkflowError(
                f"Optimization failed: {e}",
                {"stage": "optimization_orchestration", "study_name": study_name},
            ) from e

    def _report_and_record(self, study: Study, trial: Trial) -> Study:
        """Report trial result to the backend and append it to the study.

        Args:
            study: Current study state.
            trial: Trial whose outcome has just been determined.

        Returns:
            Study with the trial appended.
        """
        self._optimization_backend_session.report_trial_result(study, trial)
        return study.add_trial(trial)

    def _log_successful_trial(
        self,
        trial_context: Any | None,
        trial: Trial,
        training_result: TrainingResult,
    ) -> None:
        """Fire post-success tracking calls (metrics/artifacts/outcome/hooks); no-op if untracked.

        Args:
            trial_context: Optional tracking context for metric logging (None if untracked).
            trial: Trial that completed successfully.
            training_result: Result produced by the successful trial.
        """
        if trial_context is None:
            return
        log_trial_metrics(training_result.metrics or {}, trial_context)
        log_trial_artifacts(training_result.artifacts or {}, trial_context)
        log_trial_outcome(trial, trial_context)
        fire_post_training_hooks(self._hooks, trial_context, training_result)

    def _run_trial_iteration(
        self,
        study: Study,
        trial: Trial,
        base_settings: SearchJobConfig,
        trial_context: Any | None,
    ) -> tuple[Study, Trial, SearchJobConfig, TrainingResult | None]:
        """Execute one trial iteration through the shared tracked/untracked path.

        The caller owns study/trial tracking context managers; this method owns
        hyperparameter resolution, trial execution, trial-state updates, backend
        reporting, study mutation, and success-path logging.

        Args:
            study: Current study state
            trial: Trial to execute
            base_settings: Base configuration settings
            trial_context: Optional tracking context for metric logging (None if untracked)

        Returns:
            Updated study and trial, plus resolved trial settings and the training
            result when the trial completed successfully.
        """
        hyperparameters = self._optimization_backend_session.suggest_hyperparameters(
            study, trial, base_settings
        )
        trial_settings = self._trial_executor.apply_hyperparameters(base_settings, hyperparameters)

        if trial_context is not None:
            log_trial_settings(trial_settings, trial_context)
            log_trial_hyperparameters(hyperparameters, trial, trial_context)

        try:
            training_result = self._trial_executor.execute_trial(
                trial,
                base_settings,
                hyperparameters,
                trial_context,
            )
            objective_value = self._trial_executor.extract_objective_value(
                training_result, objective=trial_settings.search.objective
            )
            logger.info(
                "Trial {} completed with objective {}={}",
                trial.trial_number,
                trial_settings.search.objective,
                objective_value,
            )
            trial = complete_trial(
                trial,
                hyperparameters=hyperparameters,
                objective_value=objective_value,
                training_result=training_result,
            )
            study = self._report_and_record(study, trial)
            self._log_successful_trial(trial_context, trial, training_result)

            return study, trial, trial_settings, training_result

        except TrialPrunedException as e:
            trial = prune_trial(
                trial,
                hyperparameters=hyperparameters,
                pruned_at_step=e.pruned_at_step,
            )
            study = self._report_and_record(study, trial)
            return study, trial, trial_settings, None

        except TrialFailedException:
            trial = fail_trial(
                trial,
                hyperparameters=hyperparameters,
            )
            study = self._report_and_record(study, trial)
            return study, trial, trial_settings, None

    def _execute_with_tracking(
        self,
        study: Study,
        base_settings: SearchJobConfig,
    ) -> OptimizationResult:
        """Execute optimization with experiment tracking.

        This creates the proper nested MLflow run structure:
        - Parent run for the study
        - Child runs for each trial
        - Final child run for best retrain
        """
        if self._experiment_tracker is None:
            return self._execute_without_tracking(study, base_settings)
        with self._experiment_tracker.create_study_run(study) as study_context:
            # Track optimization duration
            start_time = time.time()

            # Log study metadata
            log_study_metadata(study, study_context)

            # Execute trials
            for trial_number in range(study.target_trials):
                trial = self._create_trial(study, trial_number)

                with self._experiment_tracker.create_trial_run(
                    trial, study_context
                ) as trial_context:
                    study, trial, _trial_settings, _training_result = self._run_trial_iteration(
                        study, trial, base_settings, trial_context
                    )

            # Retrain with best parameters
            best_trial = study.best_trial
            best_training_result = None
            best_settings: SearchJobConfig | None = None

            if best_trial:
                with self._experiment_tracker.create_best_retrain_run(
                    study, study_context
                ) as retrain_context:
                    logger.info(
                        "Retraining best trial {}",
                        best_trial.trial_number,
                    )

                    best_settings, best_training_result = self._retrain_best_trial(
                        base_settings, best_trial, retrain_context
                    )

            # Complete study
            study = study.complete_study()
            self._study_manager.save_study(study)

            # Create final result
            total_duration = time.time() - start_time
            result = OptimizationResult(
                study=study,
                best_trial=best_trial,
                best_training_result=best_training_result,
                total_duration_seconds=total_duration,
            )

            # Log study summary and best trial configuration
            log_study_summary(result, study_context)
            if best_trial and best_settings:
                log_best_trial_settings(best_settings, study_context)
            if best_training_result is not None:
                log_best_trial_result(best_training_result, study_context)

            return result

    def _retrain_best_trial(
        self,
        base_settings: SearchJobConfig,
        best_trial: Trial,
        retrain_context: Any,
    ) -> tuple[SearchJobConfig, TrainingResult]:
        """Retrain the best trial, using full tracking parity when a run is active.

        When ``retrain_context`` is a real (non-null) run, delegates to
        :meth:`TrialExecutor.execute_best_retrain` for ``TrackingDecorator``
        parity (checkpoints, plots, settings/model logging), then layers on
        the optimization-specific hyperparameters/outcome logging that
        ``TrackingDecorator`` doesn't know about. Otherwise falls back to the
        plain lightweight trial execution path.

        Args:
            base_settings: Base configuration settings.
            best_trial: Trial with the best objective value.
            retrain_context: Run context for the best-retrain run (may be null).

        Returns:
            Tuple of the resolved trial settings and the training result.
        """
        if not retrain_context.is_active():
            settings = self._trial_executor.apply_hyperparameters(
                base_settings, best_trial.hyperparameters
            )
            result = self._trial_executor.execute_trial(
                best_trial, base_settings, best_trial.hyperparameters
            )
            return settings, result

        if self._experiment_tracker is None:
            raise WorkflowError(
                "Best retrain run is active but no experiment tracker is configured",
                {"stage": "best_retrain_execution"},
            )

        settings, result = self._trial_executor.execute_best_retrain(
            base_settings,
            best_trial.hyperparameters,
            run_context=retrain_context,
            tracker=self._experiment_tracker.execution_tracker(),
            tracking_uri=retrain_context.tracking_uri,
            hooks=self._hooks,
        )
        # execute_best_retrain already logged model hyperparameters via
        # SettingsLogger.log_model_parameters (full TrackingDecorator path), under
        # the same stripped names log_trial_hyperparameters would use for "model."
        # keys — exclude them here to avoid logging the same param twice.
        non_model_hyperparameters = {
            key: value
            for key, value in best_trial.hyperparameters.items()
            if not key.startswith("model.")
        }
        log_trial_hyperparameters(non_model_hyperparameters, best_trial, retrain_context)
        log_trial_outcome(best_trial, retrain_context)
        return settings, result

    def _execute_without_tracking(
        self,
        study: Study,
        base_settings: SearchJobConfig,
    ) -> OptimizationResult:
        """Execute optimization without experiment tracking."""
        for trial_number in range(study.target_trials):
            trial = self._create_trial(study, trial_number)
            study, trial, _trial_settings, _training_result = self._run_trial_iteration(
                study, trial, base_settings, trial_context=None
            )

        # Retrain with best parameters
        best_trial = study.best_trial
        best_training_result = None

        if best_trial:
            best_training_result = self._trial_executor.execute_trial(
                best_trial, base_settings, best_trial.hyperparameters
            )

        # Complete study
        study = study.complete_study()
        self._study_manager.save_study(study)

        # Create final result
        total_duration = study.duration_seconds
        return OptimizationResult(
            study=study,
            best_trial=best_trial,
            best_training_result=best_training_result,
            total_duration_seconds=total_duration,
        )

    def _create_trial(self, study: Study, trial_number: int) -> Trial:
        """Create a new trial for the study.

        Args:
            study: Parent study
            trial_number: Trial number

        Returns:
            New trial instance
        """
        trial_id = f"{study.study_id}_trial_{trial_number}"
        return Trial(
            trial_id=trial_id,
            trial_number=trial_number,
            hyperparameters={},
            started_at=datetime.now(),
        )
