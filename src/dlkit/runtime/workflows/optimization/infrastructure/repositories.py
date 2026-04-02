"""Repository implementations for Study persistence using Optuna.

These repositories implement the domain repository interfaces using
Optuna as the underlying storage mechanism, following the Repository pattern.
"""

from __future__ import annotations

import uuid
from typing import Any

from dlkit.runtime.workflows.optimization.domain import (
    IStudyRepository,
    OptimizationDirection,
    Study,
    Trial,
    TrialState,
)
from dlkit.shared.errors import WorkflowError


class OptunaStudyRepository(IStudyRepository):
    """Repository implementation using Optuna as the persistence layer.

    This adapter translates between domain models (Study, Trial) and
    Optuna objects, following the Repository pattern and DIP.
    """

    def __init__(self, optuna_module: Any = None):
        """Initialize repository with optional Optuna module injection.

        Args:
            optuna_module: Optuna module for testing/DI purposes
        """
        self._optuna = optuna_module
        if self._optuna is None:
            try:
                import optuna

                self._optuna = optuna
            except ImportError as e:
                raise WorkflowError(
                    f"Optuna not available: {e}", {"stage": "repository_initialization"}
                ) from e

        # Map domain directions to Optuna directions
        self._direction_map = {
            OptimizationDirection.MINIMIZE: self._optuna.study.StudyDirection.MINIMIZE,
            OptimizationDirection.MAXIMIZE: self._optuna.study.StudyDirection.MAXIMIZE,
        }

        # Reverse mapping
        self._reverse_direction_map = {v: k for k, v in self._direction_map.items()}

        # Map Optuna trial states to domain states
        self._trial_state_map = {
            self._optuna.trial.TrialState.RUNNING: TrialState.RUNNING,
            self._optuna.trial.TrialState.COMPLETE: TrialState.COMPLETE,
            self._optuna.trial.TrialState.PRUNED: TrialState.PRUNED,
            self._optuna.trial.TrialState.FAIL: TrialState.FAILED,
        }

    def create_study(
        self,
        study_name: str,
        direction: OptimizationDirection,
        target_trials: int,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> Study:
        """Create new study using Optuna."""
        try:
            # Build sampler and pruner from configs
            sampler = self._build_sampler(sampler_config) if sampler_config else None
            pruner = self._build_pruner(pruner_config) if pruner_config else None

            # Extract storage URL from config
            storage_url = storage_config.get("url") if storage_config else None
            load_if_exists = (
                storage_config.get("load_if_exists", False) if storage_config else False
            )

            # Create Optuna study
            optuna_study = self._optuna.create_study(
                direction=self._direction_map[direction],
                sampler=sampler,
                pruner=pruner,
                study_name=study_name,
                storage=storage_url,
                load_if_exists=load_if_exists,
            )

            # Convert to domain model
            study_id = str(uuid.uuid4())  # Generate domain-specific ID
            study = Study(
                study_id=study_id,
                study_name=optuna_study.study_name,
                direction=direction,
                trials=(),
                target_trials=target_trials,
                sampler_config=sampler_config,
                pruner_config=pruner_config,
            )

            # Store mapping between domain ID and Optuna study
            self._study_mapping: dict[str, Any] = getattr(self, "_study_mapping", {})
            self._study_mapping[study_id] = optuna_study

            return study

        except Exception as e:
            raise WorkflowError(
                f"Failed to create study: {e}",
                {"stage": "study_creation", "study_name": study_name},
            ) from e

    def get_study(self, study_id: str) -> Study | None:
        """Get study by domain ID."""
        try:
            optuna_study = self._study_mapping.get(study_id)
            if not optuna_study:
                return None

            # Convert Optuna study to domain model
            direction = self._reverse_direction_map[optuna_study.direction]

            study = Study(
                study_id=study_id,
                study_name=optuna_study.study_name,
                direction=direction,
                trials=tuple(self._convert_optuna_trials(optuna_study.trials)),
            )

            return study

        except Exception as e:
            raise WorkflowError(
                f"Failed to get study: {e}", {"stage": "study_retrieval", "study_id": study_id}
            ) from e

    def save_study(self, study: Study) -> None:
        """Save study to Optuna storage."""
        # In Optuna, studies are automatically persisted when modified
        # This is a no-op for Optuna but allows other storage implementations

    def add_trial_to_study(self, study_id: str, trial: Trial) -> None:
        """Add trial to study."""
        optuna_study = self._study_mapping.get(study_id)
        if not optuna_study:
            raise WorkflowError(
                f"Study not found: {study_id}", {"stage": "trial_addition", "study_id": study_id}
            )

        # Convert domain trial to Optuna trial
        # Note: In Optuna, trials are created through the optimization process
        # This method is mainly for testing/domain consistency

    def update_trial_in_study(self, study_id: str, trial_id: str, **updates) -> None:
        """Update trial in study."""
        # In Optuna, trial updates are handled by the optimization process
        # This is mainly for domain model consistency

    def get_best_trial(self, study_id: str) -> Trial | None:
        """Get best trial from study."""
        try:
            optuna_study = self._study_mapping.get(study_id)
            if not optuna_study or not optuna_study.trials:
                return None

            best_optuna_trial = optuna_study.best_trial
            if not best_optuna_trial:
                return None

            return self._convert_optuna_trial(best_optuna_trial)

        except Exception as e:
            raise WorkflowError(
                f"Failed to get best trial: {e}",
                {"stage": "best_trial_retrieval", "study_id": study_id},
            ) from e

    def _build_sampler(self, sampler_config: dict[str, Any]) -> Any:
        """Build Optuna sampler from configuration."""
        sampler_type = sampler_config.get("type", "TPESampler")
        sampler_params = sampler_config.get("params", {})

        # Get sampler class from Optuna
        sampler_class = getattr(self._optuna.samplers, sampler_type, None)
        if not sampler_class:
            raise WorkflowError(
                f"Unknown sampler type: {sampler_type}",
                {"stage": "sampler_creation", "type": sampler_type},
            )

        return sampler_class(**sampler_params)

    def _build_pruner(self, pruner_config: dict[str, Any]) -> Any:
        """Build Optuna pruner from configuration."""
        pruner_type = pruner_config.get("type", "MedianPruner")
        pruner_params = pruner_config.get("params", {})

        # Get pruner class from Optuna
        pruner_class = getattr(self._optuna.pruners, pruner_type, None)
        if not pruner_class:
            raise WorkflowError(
                f"Unknown pruner type: {pruner_type}",
                {"stage": "pruner_creation", "type": pruner_type},
            )

        return pruner_class(**pruner_params)

    def _convert_optuna_trials(self, optuna_trials: list[Any]) -> list[Trial]:
        """Convert list of Optuna trials to domain trials."""
        return [self._convert_optuna_trial(trial) for trial in optuna_trials]

    def _convert_optuna_trial(self, optuna_trial: Any) -> Trial:
        """Convert Optuna trial to domain trial."""
        # Map Optuna trial state to domain state
        domain_state = self._trial_state_map.get(optuna_trial.state, TrialState.FAILED)

        # Extract timing information
        started_at = None
        completed_at = None
        try:
            if hasattr(optuna_trial, "datetime_start") and optuna_trial.datetime_start:
                started_at = optuna_trial.datetime_start
            if hasattr(optuna_trial, "datetime_complete") and optuna_trial.datetime_complete:
                completed_at = optuna_trial.datetime_complete
        except Exception:
            pass

        # Extract pruning information
        pruned_at_step = None
        if domain_state == TrialState.PRUNED:
            try:
                # Optuna stores intermediate values with step numbers
                if (
                    hasattr(optuna_trial, "intermediate_values")
                    and optuna_trial.intermediate_values
                ):
                    pruned_at_step = max(optuna_trial.intermediate_values.keys())
            except Exception:
                pass

        return Trial(
            trial_id=str(optuna_trial.number),  # Use trial number as ID
            trial_number=optuna_trial.number,
            hyperparameters=dict(optuna_trial.params) if hasattr(optuna_trial, "params") else {},
            objective_value=optuna_trial.value if hasattr(optuna_trial, "value") else None,
            state=domain_state,
            started_at=started_at,
            completed_at=completed_at,
            pruned_at_step=pruned_at_step,
        )


class InMemoryStudyRepository(IStudyRepository):
    """In-memory repository implementation for testing.

    This implementation stores studies in memory without any external dependencies,
    making it ideal for unit testing and development.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._studies: dict[str, Study] = {}

    def create_study(
        self,
        study_name: str,
        direction: OptimizationDirection,
        target_trials: int,
        sampler_config: dict[str, Any] | None = None,
        pruner_config: dict[str, Any] | None = None,
        storage_config: dict[str, Any] | None = None,
    ) -> Study:
        """Create new study in memory."""
        study_id = str(uuid.uuid4())
        study = Study(
            study_id=study_id,
            study_name=study_name,
            direction=direction,
            trials=(),
            target_trials=target_trials,
            sampler_config=sampler_config,
            pruner_config=pruner_config,
        )

        self._studies[study_id] = study
        return study

    def get_study(self, study_id: str) -> Study | None:
        """Get study from memory."""
        return self._studies.get(study_id)

    def save_study(self, study: Study) -> None:
        """Save study to memory."""
        self._studies[study.study_id] = study

    def add_trial_to_study(self, study_id: str, trial: Trial) -> None:
        """Add trial to study in memory."""
        study = self._studies.get(study_id)
        if not study:
            raise WorkflowError(
                f"Study not found: {study_id}", {"stage": "trial_addition", "study_id": study_id}
            )

        self._studies[study_id] = study.add_trial(trial)

    def update_trial_in_study(self, study_id: str, trial_id: str, **updates) -> None:
        """Update trial in study in memory."""
        study = self._studies.get(study_id)
        if not study:
            raise WorkflowError(
                f"Study not found: {study_id}", {"stage": "trial_update", "study_id": study_id}
            )

        self._studies[study_id] = study.update_trial(trial_id, **updates)

    def get_best_trial(self, study_id: str) -> Trial | None:
        """Get best trial from study in memory."""
        study = self._studies.get(study_id)
        return study.best_trial if study else None
