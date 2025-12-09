"""Application services for optimization orchestration following SOLID principles.

These services coordinate between domain models, repositories, and infrastructure
adapters to execute optimization workflows. Each service has a single responsibility
and depends on abstractions rather than concrete implementations.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from dlkit.interfaces.api.domain import TrainingResult, WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import (
    TrainingWorkflowConfig,
    OptimizationWorkflowConfig,
)
from dlkit.tools.utils.logging_config import get_logger
from dlkit.runtime.workflows.factories.build_factory import BuildFactory, BuildComponents

from ..domain import (
    Study,
    Trial,
    OptimizationResult,
    OptimizationDirection,
    TrialState,
    IStudyRepository,
    IExperimentTracker,
    IConfigurationPersistence,
    TrialPrunedException,
    TrialFailedException,
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
            "Creating optimization study",
            study_name=study_name,
            direction=direction.value,
            target_trials=target_trials,
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
                "Study created successfully",
                study_id=study.study_id,
                study_name=study.study_name,
            )

            return study

        except Exception as e:
            logger.error("Failed to create study", error=str(e))
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

        study.complete_study()
        self._repository.save_study(study)

        logger.info(
            "Study completed",
            study_id=study_id,
            total_trials=len(study.trials),
            successful_trials=len(study.successful_trials),
        )


class TrialExecutor:
    """Service responsible for executing individual optimization trials.

    This service handles the execution of single trials, including hyperparameter
    sampling, training execution, and result collection.
    """

    def __init__(self, build_factory: BuildFactory):
        """Initialize trial executor.

        Args:
            build_factory: Factory for building training components
        """
        self._build_factory = build_factory

    def execute_trial(
        self,
        trial: Trial,
        base_settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        hyperparameters: dict[str, Any],
        trial_context: Any = None,
        enable_checkpointing: bool = False,
    ) -> TrainingResult:
        """Execute a single optimization trial.

        Args:
            trial: Trial domain model
            base_settings: Base configuration settings
            hyperparameters: Hyperparameters for this trial
            trial_context: Optional trial run context for metric logging
            enable_checkpointing: Whether to enable checkpointing (default False)

        Returns:
            Training result from trial execution

        Raises:
            TrialPrunedException: If trial should be pruned
            TrialFailedException: If trial execution fails
        """
        logger.info(
            "Executing optimization trial",
            trial_id=trial.trial_id,
            trial_number=trial.trial_number,
            hyperparameters=hyperparameters,
        )

        try:
            # Apply hyperparameters to base settings
            trial_settings = self._apply_hyperparameters(base_settings, hyperparameters)

            # Build components for this trial
            components = self._build_factory.build_components(trial_settings)

            # Execute training with optional metric logging and checkpoint control
            # TODO: Add pruning callback injection here
            training_result = self._execute_training(
                components, trial_settings, trial_context, enable_checkpointing
            )

            logger.info(
                "Trial executed successfully",
                trial_id=trial.trial_id,
                trial_number=trial.trial_number,
                objective_value=self._extract_objective_value(training_result),
            )

            return training_result

        except TrialPrunedException:
            logger.info(
                "Trial pruned",
                trial_id=trial.trial_id,
                trial_number=trial.trial_number,
            )
            raise

        except Exception as e:
            logger.error(
                "Trial execution failed",
                trial_id=trial.trial_id,
                trial_number=trial.trial_number,
                error=str(e),
            )
            raise TrialFailedException(f"Trial execution failed: {e}") from e

    def _apply_hyperparameters(
        self,
        base_settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        hyperparameters: dict[str, Any],
    ) -> GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig:
        """Apply hyperparameters to base settings.

        Args:
            base_settings: Base configuration
            hyperparameters: Hyperparameters to apply

        Returns:
            Settings with hyperparameters applied
        """
        # TODO: Implement proper hyperparameter application
        # This should use the existing settings sampler logic
        try:
            if base_settings.MODEL and hyperparameters:
                # Apply hyperparameters to model settings
                updated_model = base_settings.MODEL.model_copy(update=hyperparameters)
                return base_settings.model_copy(update={"MODEL": updated_model})
        except Exception as e:
            logger.warning("Failed to apply hyperparameters", error=str(e))

        return base_settings

    def _execute_training(
        self,
        components: BuildComponents,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
        trial_context: Any = None,
        enable_checkpointing: bool = False,
    ) -> TrainingResult:
        """Execute training with given components and optional checkpoint control.

        Args:
            components: Built training components
            settings: Training settings
            trial_context: Optional trial run context for metric logging
            enable_checkpointing: Whether to enable checkpointing (default False for trials)

        Returns:
            Training result
        """
        # Disable checkpointing for optimization trials (only enable for best model)
        if not enable_checkpointing:
            self._disable_checkpoints(components)

        # Inject MLflow epoch logger if trial context is provided
        if trial_context is not None:
            self._inject_mlflow_logger(components, trial_context)

        # Use existing VanillaExecutor for actual training
        from dlkit.runtime.workflows.strategies.core import VanillaExecutor

        executor = VanillaExecutor()
        return executor.execute(components, settings)

    def _disable_checkpoints(self, components: BuildComponents) -> None:
        """Remove checkpoint callbacks from trainer (SRP: single responsibility).

        During optimization, we don't want to save checkpoints for every trial
        as they take up disk space. We only checkpoint the final best model.

        Args:
            components: Build components with trainer
        """
        from pytorch_lightning.callbacks import ModelCheckpoint

        trainer = getattr(components, "trainer", None)
        if not trainer or not hasattr(trainer, "callbacks"):
            return

        try:
            original_count = len(trainer.callbacks)
            trainer.callbacks = [
                cb for cb in trainer.callbacks if not isinstance(cb, ModelCheckpoint)
            ]
            removed_count = original_count - len(trainer.callbacks)

            if removed_count > 0:
                logger.debug(
                    f"Disabled {removed_count} checkpoint callback(s) for optimization trial"
                )

        except Exception as e:
            logger.warning(f"Failed to disable checkpoints: {e}")

    def _inject_mlflow_logger(self, components: BuildComponents, trial_context: Any) -> None:
        """Inject MLflow epoch logger callback for metric logging during optimization.

        Args:
            components: Build components
            trial_context: Trial run context with run_id for logging
        """
        try:
            trainer = getattr(components, "trainer", None)
            if not trainer:
                return

            # Create callback that logs metrics with epoch numbers instead of steps
            from dlkit.core.training.callbacks import MLflowEpochLogger

            # The trial_context from MLflowTrackingAdapter wraps a run_context
            # We need to pass the underlying run_context to the callback
            run_context = getattr(trial_context, "_run_context", trial_context)

            epoch_logger = MLflowEpochLogger(run_context)

            # Add callback to trainer
            if not hasattr(trainer, "callbacks"):
                trainer.callbacks = []
            trainer.callbacks.append(epoch_logger)
            logger.debug(
                f"Injected MLflowEpochLogger callback for optimization trial (run_id={getattr(run_context, 'run_id', 'unknown')})"
            )

        except Exception as e:
            logger.warning(f"Failed to inject MLflow epoch logger for trial: {e}")

    def _extract_objective_value(self, training_result: TrainingResult) -> float:
        """Extract objective value from training result.

        Args:
            training_result: Result from training

        Returns:
            Objective value for optimization
        """
        if not training_result.metrics:
            return 0.0

        # Try common objective metric names
        for key in ["val_loss", "valid_loss", "loss", "train_loss"]:
            if key in training_result.metrics:
                try:
                    return float(training_result.metrics[key])
                except (ValueError, TypeError):
                    continue

        return 0.0


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
        experiment_tracker: IExperimentTracker | None = None,
        config_persister: IConfigurationPersistence | None = None,
    ):
        """Initialize optimization orchestrator.

        Args:
            study_manager: Study lifecycle management service
            trial_executor: Trial execution service
            experiment_tracker: Optional experiment tracking
            config_persister: Optional configuration persistence
        """
        self._study_manager = study_manager
        self._trial_executor = trial_executor
        self._experiment_tracker = experiment_tracker
        self._config_persister = config_persister

    def execute_optimization(
        self,
        study_name: str,
        base_settings: GeneralSettings | OptimizationWorkflowConfig,
        n_trials: int,
        direction: OptimizationDirection,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Execute complete optimization workflow.

        IMPORTANT: This method manages the experiment tracker context lifecycle.
        The tracker context is entered RIGHT BEFORE optimization work begins and
        exited RIGHT AFTER work completes, ensuring server lifetime matches work duration.

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
            "Starting optimization workflow",
            study_name=study_name,
            n_trials=n_trials,
            direction=direction.value,
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

            if tracker is not None:
                with tracker:
                    study = self._study_manager.create_study(**study_kwargs)
                    result = self._execute_with_tracking(study, base_settings)
                return result

            study = self._study_manager.create_study(**study_kwargs)
            return self._execute_without_tracking(study, base_settings)

        except Exception as e:
            logger.error("Optimization workflow failed", error=str(e))
            raise WorkflowError(
                f"Optimization failed: {e}",
                {"stage": "optimization_orchestration", "study_name": study_name},
            ) from e

    def _execute_with_tracking(
        self, study: Study, base_settings: GeneralSettings | OptimizationWorkflowConfig
    ) -> OptimizationResult:
        """Execute optimization with experiment tracking.

        This creates the proper nested MLflow run structure:
        - Parent run for the study
        - Child runs for each trial
        - Final child run for best retrain
        """
        with self._experiment_tracker.create_study_run(study) as study_context:
            # Track optimization duration
            start_time = time.time()

            # Log study metadata
            study_context.log_study_metadata(study)

            # Execute trials
            for trial_number in range(study.target_trials):
                trial = self._create_trial(study, trial_number)

                with self._experiment_tracker.create_trial_run(
                    trial, study_context
                ) as trial_context:
                    try:
                        # Sample hyperparameters BEFORE training
                        hyperparameters = self._sample_hyperparameters(trial, study, base_settings)
                        trial_settings = self._trial_executor._apply_hyperparameters(
                            base_settings, hyperparameters
                        )

                        # Log trial configuration BEFORE training (hyperparameters must be logged at start!)
                        trial_context.log_trial_settings(trial_settings)
                        # Only log sampled hyperparameters (not model_ prefixed duplicates)
                        trial_context.log_trial_hyperparameters(hyperparameters)

                        # Execute trial with trial context for metric logging
                        # Disable checkpointing for exploratory trials (enable_checkpointing=False)
                        training_result = self._trial_executor.execute_trial(
                            trial, base_settings, hyperparameters, trial_context, enable_checkpointing=False
                        )

                        # Update trial with results
                        objective_value = self._trial_executor._extract_objective_value(training_result)
                        trial = trial.__class__(**{
                            **trial.__dict__,
                            "objective_value": objective_value,
                            "training_result": training_result,
                            "state": TrialState.COMPLETE,
                            "completed_at": datetime.now(),
                        })

                        # Report result back to Optuna study using study.tell()
                        self._report_trial_to_optuna(trial, objective_value, study)

                        # Log trial results AFTER training
                        trial_context.log_trial_metrics(training_result.metrics or {})
                        trial_context.log_trial_artifacts(training_result.artifacts or {})

                        # Add trial to study
                        study.add_trial(trial)

                    except TrialPrunedException as e:
                        # Handle pruned trial
                        trial = trial.__class__(**{
                            **trial.__dict__,
                            "state": TrialState.PRUNED,
                            "pruned_at_step": e.pruned_at_step,
                            "completed_at": datetime.now(),
                        })
                        # Report pruned trial to Optuna
                        self._report_trial_to_optuna(trial, None, study, state="pruned")
                        study.add_trial(trial)

                    except TrialFailedException:
                        # Handle failed trial
                        trial = trial.__class__(**{
                            **trial.__dict__,
                            "state": TrialState.FAILED,
                            "completed_at": datetime.now(),
                        })
                        # Report failed trial to Optuna
                        self._report_trial_to_optuna(trial, None, study, state="fail")
                        study.add_trial(trial)

            # Retrain with best parameters
            best_trial = study.best_trial
            best_training_result = None

            if best_trial:
                with self._experiment_tracker.create_best_retrain_run(
                    study, study_context
                ) as retrain_context:
                    logger.info(
                        "Retraining with best hyperparameters",
                        best_trial_number=best_trial.trial_number,
                        best_hyperparameters=best_trial.hyperparameters,
                    )

                    # Execute best retrain with retrain context for metric logging
                    best_settings = self._trial_executor._apply_hyperparameters(
                        base_settings, best_trial.hyperparameters
                    )
                    # Enable checkpointing for best model retraining
                    best_training_result = self._trial_executor.execute_trial(
                        best_trial, base_settings, best_trial.hyperparameters, retrain_context, enable_checkpointing=True
                    )

                    # Log best retrain configuration and results
                    retrain_context.log_trial_settings(best_settings)
                    retrain_context.log_model_hyperparameters(best_settings)
                    retrain_context.log_trial_hyperparameters(best_trial.hyperparameters)
                    retrain_context.log_trial_metrics(best_training_result.metrics or {})
                    retrain_context.log_trial_artifacts(best_training_result.artifacts or {})

            # Complete study
            study.complete_study()
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
            study_context.log_study_summary(result)
            if best_trial and best_settings:
                study_context.log_best_trial_settings(best_settings)

            return result

    def _execute_without_tracking(
        self, study: Study, base_settings: GeneralSettings | OptimizationWorkflowConfig
    ) -> OptimizationResult:
        """Execute optimization without experiment tracking."""
        # Similar logic but without tracking context managers
        # This is a simplified version for when tracking is disabled

        for trial_number in range(study.target_trials):
            trial = self._create_trial(study, trial_number)

            try:
                hyperparameters = self._sample_hyperparameters(trial, study, base_settings)
                training_result = self._trial_executor.execute_trial(
                    trial, base_settings, hyperparameters
                )

                # Update trial with results
                trial = trial.__class__(**{
                    **trial.__dict__,
                    "objective_value": self._trial_executor._extract_objective_value(
                        training_result
                    ),
                    "training_result": training_result,
                    "state": TrialState.COMPLETE,
                    "completed_at": datetime.now(),
                })

                study.add_trial(trial)

            except (TrialPrunedException, TrialFailedException):
                # Handle failed/pruned trials
                trial = trial.__class__(**{
                    **trial.__dict__,
                    "state": TrialState.FAILED,
                    "completed_at": datetime.now(),
                })
                study.add_trial(trial)

        # Retrain with best parameters
        best_trial = study.best_trial
        best_training_result = None

        if best_trial:
            best_training_result = self._trial_executor.execute_trial(
                best_trial, base_settings, best_trial.hyperparameters
            )

        # Complete study
        study.complete_study()
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

    def _sample_hyperparameters(
        self, trial: Trial, study: Study, base_settings: GeneralSettings | OptimizationWorkflowConfig
    ) -> dict[str, Any]:
        """Sample hyperparameters for a trial using Optuna's suggest methods.

        Args:
            trial: Trial to sample for
            study: Parent study
            base_settings: Base configuration settings

        Returns:
            Sampled hyperparameters
        """
        # Get the actual Optuna study from repository via study_manager
        optuna_study = None
        if hasattr(self._study_manager, '_repository'):
            repo = self._study_manager._repository
            if hasattr(repo, '_study_mapping'):
                optuna_study = repo._study_mapping.get(study.study_id)

        # If we don't have an Optuna study, we can't sample - return empty dict
        if not optuna_study:
            logger.warning("No Optuna study available for sampling, using base settings")
            return {}

        # Get OPTUNA configuration from base settings
        optuna_config = getattr(base_settings, "OPTUNA", None)
        if not optuna_config or not getattr(optuna_config, "enabled", False):
            logger.warning("OPTUNA not enabled in settings, using base settings")
            return {}

        # Use Optuna's study.ask() to get a trial, then sample from it
        try:
            optuna_trial = optuna_study.ask()

            # Use the OptunaSettingsSampler to sample hyperparameters
            from dlkit.tools.config.samplers.optuna_sampler import create_settings_sampler

            settings_sampler = create_settings_sampler(optuna_config)
            sampled_settings = settings_sampler.sample(optuna_trial, base_settings)

            # Extract ALL sampled hyperparameters from optuna_trial.params
            # These are the actual sampled values from Optuna's suggest methods
            hyperparameters = dict(optuna_trial.params)

            # Store the optuna trial in a mapping for later reporting with study.tell()
            if not hasattr(self, '_optuna_trials'):
                self._optuna_trials = {}
            self._optuna_trials[trial.trial_id] = optuna_trial

            logger.debug(f"Sampled hyperparameters from Optuna: {hyperparameters}")
            return hyperparameters

        except Exception as e:
            logger.warning(f"Failed to sample hyperparameters from Optuna: {e}", exc_info=True)

        return {}

    def _report_trial_to_optuna(
        self, trial: Trial, objective_value: float | None, study: Study, state: str = "complete"
    ) -> None:
        """Report trial results back to Optuna using study.tell().

        Args:
            trial: Domain trial object
            objective_value: Objective value to report (None for failed/pruned)
            study: Parent study
            state: Trial state ("complete", "fail", "pruned")
        """
        # Get the Optuna trial and study
        optuna_trial = getattr(self, '_optuna_trials', {}).get(trial.trial_id)
        if not optuna_trial:
            logger.warning(f"No Optuna trial found for {trial.trial_id}, skipping study.tell()")
            return

        optuna_study = None
        if hasattr(self._study_manager, '_repository'):
            repo = self._study_manager._repository
            if hasattr(repo, '_study_mapping'):
                optuna_study = repo._study_mapping.get(study.study_id)

        if not optuna_study:
            logger.warning(f"No Optuna study found for {study.study_id}, skipping study.tell()")
            return

        try:
            # Import optuna trial states
            import optuna

            # Map state to Optuna state
            state_map = {
                "complete": optuna.trial.TrialState.COMPLETE,
                "fail": optuna.trial.TrialState.FAIL,
                "pruned": optuna.trial.TrialState.PRUNED,
            }
            optuna_state = state_map.get(state, optuna.trial.TrialState.COMPLETE)

            # Report to Optuna
            optuna_study.tell(optuna_trial, objective_value, state=optuna_state)
            logger.debug(
                f"Reported trial {trial.trial_number} to Optuna: value={objective_value}, state={state}"
            )

        except Exception as e:
            logger.warning(f"Failed to report trial to Optuna: {e}")
