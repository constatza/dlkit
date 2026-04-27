"""Domain protocols for optimization following DIP and ABC patterns.

These protocols define the core abstractions that the domain layer depends on,
following the Dependency Inversion Principle. All concrete implementations
in the infrastructure layer implement these protocols.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable

from .models import OptimizationDirection, OptimizationResult, Study, Trial


class IStudyRepository(ABC):
    """Repository abstraction for Study persistence following DIP.

    This abstraction allows the domain layer to persist studies without
    depending on concrete storage implementations like Optuna.
    """

    @abstractmethod
    def create_study(
        self,
        study_name: str,
        direction: OptimizationDirection,
        target_trials: int,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> Study:
        """Create a new study.

        Args:
            study_name: Name of the study
            direction: Optimization direction
            target_trials: Number of trials to run
            sampler_config: Sampler configuration
            pruner_config: Pruner configuration
            storage_config: Storage configuration

        Returns:
            Created study domain model
        """
        raise NotImplementedError

    @abstractmethod
    def get_study(self, study_id: str) -> Study | None:
        """Get study by ID.

        Args:
            study_id: Study identifier

        Returns:
            Study if found, None otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def save_study(self, study: Study) -> None:
        """Save study to persistence layer.

        Args:
            study: Study to save
        """
        raise NotImplementedError

    @abstractmethod
    def add_trial_to_study(self, study_id: str, trial: Trial) -> None:
        """Add trial to existing study.

        Args:
            study_id: Study identifier
            trial: Trial to add
        """
        raise NotImplementedError

    @abstractmethod
    def update_trial_in_study(self, study_id: str, trial_id: str, **updates) -> None:
        """Update trial in study.

        Args:
            study_id: Study identifier
            trial_id: Trial identifier
            updates: Trial updates to apply
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_trial(self, study_id: str) -> Trial | None:
        """Get best trial from study.

        Args:
            study_id: Study identifier

        Returns:
            Best trial if found, None otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def get_optuna_study(self, study_id: str) -> Any:
        """Return the underlying Optuna study object for a domain study ID.

        Args:
            study_id: Domain study identifier

        Returns:
            Optuna study object

        Raises:
            WorkflowError: If Optuna study not found for domain study
        """
        raise NotImplementedError


class IHyperparameterSampler(ABC):
    """Abstraction for hyperparameter sampling strategies.

    This allows different sampling strategies (random, TPE, etc.) to be used
    without the domain layer depending on specific implementations.
    """

    @abstractmethod
    def suggest_hyperparameters(
        self, trial_number: int, study_context: dict[str, Any]
    ) -> dict[str, Any]:
        """Suggest hyperparameters for a trial.

        Args:
            trial_number: Current trial number
            study_context: Study context with parameter space definition

        Returns:
            Suggested hyperparameters
        """
        raise NotImplementedError


class IPruningStrategy(ABC):
    """Abstraction for trial pruning strategies.

    This allows different pruning algorithms (median, percentile, etc.)
    to be used without coupling to specific implementations.
    """

    @abstractmethod
    def should_prune(
        self,
        trial_number: int,
        step: int,
        intermediate_value: float,
        study_context: dict[str, Any],
    ) -> bool:
        """Determine if trial should be pruned.

        Args:
            trial_number: Current trial number
            step: Current training step
            intermediate_value: Current objective value
            study_context: Study context with pruning history

        Returns:
            True if trial should be pruned
        """
        raise NotImplementedError


class IExperimentTracker(AbstractContextManager, ABC):
    """Abstraction for experiment tracking systems with guaranteed context management.

    This allows different tracking systems (MLflow, W&B, etc.) to be used
    without the domain layer depending on specific implementations.

    All experiment trackers MUST implement context manager protocol (__enter__/__exit__)
    to ensure proper resource lifecycle management. The tracker context should be
    entered BEFORE calling create_study_run() or create_trial_run(). Use no-op
    implementations for trackers that don't require resource setup/cleanup.
    """

    @abstractmethod
    def create_study_run(self, study: Study) -> AbstractContextManager[IStudyRunContext]:
        """Create a parent run for the entire optimization study.

        Args:
            study: Study domain model

        Returns:
            Context manager for study run
        """
        raise NotImplementedError

    @abstractmethod
    def create_trial_run(
        self, trial: Trial, parent_context: IStudyRunContext
    ) -> AbstractContextManager[ITrialRunContext]:
        """Create a nested run for a single trial.

        Args:
            trial: Trial domain model
            parent_context: Parent study run context

        Returns:
            Context manager for trial run
        """
        raise NotImplementedError

    @abstractmethod
    def create_best_retrain_run(
        self, study: Study, parent_context: IStudyRunContext
    ) -> AbstractContextManager[ITrialRunContext]:
        """Create a nested run for best parameter retraining.

        Args:
            study: Study with best trial information
            parent_context: Parent study run context

        Returns:
            Context manager for best retrain run
        """
        raise NotImplementedError


class IStudyRunContext(ABC):
    """Context for study-level experiment tracking."""

    @abstractmethod
    def log_study_metadata(self, study: Study) -> None:
        """Log study-level metadata.

        Args:
            study: Study domain model
        """
        raise NotImplementedError

    @abstractmethod
    def log_study_summary(self, result: OptimizationResult) -> None:
        """Log final study summary.

        Args:
            result: Optimization result
        """
        raise NotImplementedError

    @abstractmethod
    def log_best_trial_settings(self, settings: Any) -> None:
        """Log best trial settings as TOML artifact.

        Args:
            settings: GeneralSettings object for the best trial
        """
        raise NotImplementedError


class ITrialRunContext(ABC):
    """Context for trial-level experiment tracking."""

    @abstractmethod
    def log_trial_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Log trial hyperparameters.

        Args:
            hyperparameters: Trial hyperparameters
        """
        raise NotImplementedError

    @abstractmethod
    def log_trial_metrics(self, metrics: dict[str, Any]) -> None:
        """Log trial metrics.

        Args:
            metrics: Trial metrics
        """
        raise NotImplementedError

    @abstractmethod
    def log_trial_artifacts(self, artifacts: dict[str, Any]) -> None:
        """Log trial artifacts.

        Args:
            artifacts: Trial artifacts
        """
        raise NotImplementedError

    @abstractmethod
    def log_trial_settings(self, settings: Any) -> None:
        """Log complete trial settings as TOML artifact.

        Args:
            settings: GeneralSettings object for this trial
        """
        raise NotImplementedError

    @abstractmethod
    def log_model_hyperparameters(self, settings: Any) -> None:
        """Log model hyperparameters from settings.MODEL.

        Args:
            settings: GeneralSettings object with MODEL configuration
        """
        raise NotImplementedError


class IConfigurationPersistence(ABC):
    """Abstraction for configuration file persistence.

    This allows different configuration storage formats (TOML, YAML, JSON)
    to be used without coupling to specific implementations.
    """

    @abstractmethod
    def save_best_configuration(self, study: Study, configuration: dict[str, Any]) -> str | None:
        """Save best configuration to file.

        Args:
            study: Study domain model
            configuration: Configuration to save

        Returns:
            Path to saved configuration file if successful, None otherwise
        """
        raise NotImplementedError


class ITrialExecutor(ABC):
    """Abstraction for executing individual optimization trials.

    This defines how individual trials are executed without depending
    on specific training frameworks or execution strategies.
    """

    @abstractmethod
    def execute_trial(self, trial: Trial, hyperparameters: dict[str, Any]) -> float:
        """Execute a single optimization trial.

        Args:
            trial: Trial domain model
            hyperparameters: Hyperparameters for this trial

        Returns:
            Objective value for the trial

        Raises:
            TrialPrunedException: If trial should be pruned
            TrialFailedException: If trial execution fails
        """
        raise NotImplementedError


@runtime_checkable
class IHyperparameterApplicator(Protocol):
    """Protocol for applying sampled hyperparameters to workflow settings.

    Using Protocol instead of ABC for lightweight structural typing
    that doesn't require inheritance.
    """

    def apply(
        self,
        base_settings: Any,
        hyperparameters: dict[str, Any],
    ) -> Any:
        """Apply sampled hyperparameters to base settings.

        Args:
            base_settings: Base workflow settings
            hyperparameters: Sampled hyperparameters to apply

        Returns:
            Updated settings with hyperparameters applied
        """
        ...


@runtime_checkable
class IObjectiveFunction(Protocol):
    """Protocol for optimization objective functions.

    Using Protocol instead of ABC for lightweight interface definition
    that doesn't require inheritance.
    """

    def __call__(self, hyperparameters: dict[str, Any]) -> float:
        """Evaluate objective function with given hyperparameters.

        Args:
            hyperparameters: Hyperparameters to evaluate

        Returns:
            Objective value
        """
        ...
