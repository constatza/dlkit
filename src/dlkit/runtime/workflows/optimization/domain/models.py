"""Pure domain models for hyperparameter optimization following DDD principles.

These models represent the core business concepts without any infrastructure dependencies.
They model the proper Optuna Study → Trial hierarchy for clean architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any

from dlkit.domain import TrainingResult


class OptimizationDirection(Enum):
    """Optimization direction enumeration."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class TrialState(Enum):
    """Trial execution state."""

    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAILED = "FAIL"


@dataclass(frozen=True, slots=True, kw_only=True)
class HyperParameter:
    """Single hyperparameter definition."""

    name: str
    value: Any
    parameter_type: str  # 'categorical', 'uniform', 'int', etc.


@dataclass(frozen=True, slots=True, kw_only=True)
class Trial:
    """Domain model representing a single optimization trial.

    Represents one attempt at finding optimal hyperparameters within a Study.
    This is a pure domain model with no infrastructure dependencies.
    """

    trial_id: str
    trial_number: int
    hyperparameters: dict[str, Any]
    objective_value: float | None = None
    state: TrialState = TrialState.RUNNING
    training_result: TrainingResult | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    pruned_at_step: int | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate trial duration in seconds."""
        if not self.started_at:
            return 0.0
        end_time = self.completed_at or datetime.now()
        return (end_time - self.started_at).total_seconds()

    @property
    def is_complete(self) -> bool:
        """Check if trial completed successfully."""
        return self.state == TrialState.COMPLETE and self.objective_value is not None

    @property
    def is_failed(self) -> bool:
        """Check if trial failed."""
        return self.state == TrialState.FAILED

    @property
    def is_pruned(self) -> bool:
        """Check if trial was pruned."""
        return self.state == TrialState.PRUNED


@dataclass(frozen=True, slots=True, kw_only=True)
class Study:
    """Domain model representing an optimization study.

    A Study is the aggregate root that contains multiple Trials.
    This models the core Optuna concept where a Study optimizes an objective
    function through multiple Trial attempts.
    """

    study_id: str
    study_name: str
    direction: OptimizationDirection
    trials: tuple[Trial, ...] = field(default_factory=tuple)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    target_trials: int = 100
    pruner_config: dict[str, Any] | None = None
    sampler_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trials", tuple(self.trials))

    @property
    def is_complete(self) -> bool:
        """Check if study has completed all intended trials."""
        return len(self.trials) >= self.target_trials

    @property
    def successful_trials(self) -> list[Trial]:
        """Get all successfully completed trials."""
        return [t for t in self.trials if t.is_complete]

    @property
    def best_trial(self) -> Trial | None:
        """Get the best trial based on optimization direction."""
        successful = self.successful_trials
        if not successful:
            return None

        if self.direction == OptimizationDirection.MINIMIZE:
            return min(successful, key=lambda t: t.objective_value or float("inf"))
        return max(successful, key=lambda t: t.objective_value or float("-inf"))

    @property
    def best_hyperparameters(self) -> dict[str, Any]:
        """Get hyperparameters from best trial."""
        best = self.best_trial
        return best.hyperparameters if best else {}

    @property
    def best_objective_value(self) -> float | None:
        """Get objective value from best trial."""
        best = self.best_trial
        return best.objective_value if best else None

    @property
    def duration_seconds(self) -> float:
        """Calculate total study duration."""
        if not self.completed_at:
            return (datetime.now() - self.created_at).total_seconds()
        return (self.completed_at - self.created_at).total_seconds()

    def add_trial(self, trial: Trial) -> Study:
        """Add a new trial to the study."""
        if any(t.trial_id == trial.trial_id for t in self.trials):
            raise ValueError(f"Trial {trial.trial_id} already exists in study")
        return replace(self, trials=self.trials + (trial,))

    def update_trial(self, trial_id: str, **updates) -> Study:
        """Update an existing trial with new information."""
        for idx, trial in enumerate(self.trials):
            if trial.trial_id == trial_id:
                updated_trial = replace(trial, **updates)
                updated_trials = list(self.trials)
                updated_trials[idx] = updated_trial
                return replace(self, trials=tuple(updated_trials))
        raise ValueError(f"Trial {trial_id} not found in study")

    def get_trial(self, trial_id: str) -> Trial | None:
        """Get trial by ID."""
        for trial in self.trials:
            if trial.trial_id == trial_id:
                return trial
        return None

    def complete_study(self) -> Study:
        """Mark study as completed."""
        return replace(self, completed_at=datetime.now())


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizationResult:
    """Domain model for optimization results.

    Clean result object that contains the complete study outcome
    including the best trial and overall study metadata.
    """

    study: Study
    best_trial: Trial | None
    best_training_result: TrainingResult | None
    total_duration_seconds: float

    @property
    def best_hyperparameters(self) -> dict[str, Any]:
        """Get best hyperparameters from study."""
        return self.study.best_hyperparameters

    @property
    def best_objective_value(self) -> float | None:
        """Get best objective value from study."""
        return self.study.best_objective_value

    @property
    def total_trials(self) -> int:
        """Get total number of trials in study."""
        return len(self.study.trials)

    @property
    def successful_trials(self) -> int:
        """Get number of successful trials."""
        return len(self.study.successful_trials)

    @property
    def study_summary(self) -> dict[str, Any]:
        """Get summary of the optimization study."""
        return {
            "study_id": self.study.study_id,
            "study_name": self.study.study_name,
            "direction": self.study.direction.value,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "best_objective_value": self.best_objective_value,
            "duration_seconds": self.total_duration_seconds,
            "created_at": self.study.created_at.isoformat(),
            "completed_at": self.study.completed_at.isoformat()
            if self.study.completed_at
            else None,
        }


class ITrialObjective(ABC):
    """Abstract interface for trial objective functions.

    This interface defines how individual trials are executed and evaluated.
    It follows the DIP by depending on abstractions rather than concrete implementations.
    """

    @abstractmethod
    def evaluate_trial(self, trial: Trial, hyperparameters: dict[str, Any]) -> float:
        """Evaluate a single trial with given hyperparameters.

        Args:
            trial: The trial being evaluated
            hyperparameters: Hyperparameters to use for this trial

        Returns:
            Objective value for the trial

        Raises:
            TrialPrunedException: If trial should be pruned
            TrialFailedException: If trial execution fails
        """
        raise NotImplementedError


class TrialPrunedException(Exception):
    """Exception raised when a trial should be pruned."""

    def __init__(self, message: str, pruned_at_step: int):
        super().__init__(message)
        self.pruned_at_step = pruned_at_step


class TrialFailedException(Exception):
    """Exception raised when a trial execution fails."""
