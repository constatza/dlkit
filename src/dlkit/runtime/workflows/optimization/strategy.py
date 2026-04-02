"""Optimization strategy implementation using clean architecture.

This strategy implements IOptimizationStrategy and serves as the integration
point between the old strategy interface and the new clean architecture.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from dlkit.runtime.execution.interfaces import IOptimizationStrategy
from dlkit.shared import OptimizationResult as APIOptimizationResult
from dlkit.shared.errors import WorkflowError
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.workflow_configs import OptimizationWorkflowConfig, TrainingWorkflowConfig
from dlkit.tools.utils.logging_config import get_logger

if TYPE_CHECKING:
    from .factory import OptimizationServiceFactory

logger = get_logger(__name__)


class OptimizationStrategy(IOptimizationStrategy):
    """Clean optimization strategy implementing IOptimizationStrategy interface.

    This strategy serves as a bridge between the old interface and the new
    clean domain-driven architecture. It uses dependency injection to create
    and coordinate optimization services.
    """

    def __init__(
        self,
        factory: OptimizationServiceFactory,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ):
        """Initialize optimization strategy.

        Args:
            factory: Service factory for dependency injection
            settings: Configuration settings
        """
        self._factory = factory
        self._settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig = (
            settings
        )

    def execute_optimization(
        self,
        settings: GeneralSettings | TrainingWorkflowConfig | OptimizationWorkflowConfig,
    ) -> APIOptimizationResult:
        """Execute optimization using clean architecture.

        The orchestrator now manages the experiment tracker lifecycle, so the
        caller does not need to enter tracker contexts explicitly.

        Args:
            settings: Configuration settings with optimization parameters

        Returns:
            Optimization result compatible with existing API

        Raises:
            WorkflowError: If optimization fails
        """
        logger.debug("Executing optimization with clean architecture")
        start_time = time.time()

        try:
            # Create orchestrator with dependency injection
            orchestrator = self._factory.create_optimization_orchestrator(settings)

            # Extract optimization configuration
            config = self._factory.extract_optimization_config(settings)

            # Execute optimization workflow - orchestrator owns tracker context
            domain_result = orchestrator.execute_optimization(
                study_name=config["study_name"],
                base_settings=settings,
                n_trials=config["n_trials"],
                direction=config["direction"],
                sampler_config=config.get("sampler_config"),
                pruner_config=config.get("pruner_config"),
                storage_config=config.get("storage_config"),
            )

            # Convert domain result to API result for compatibility
            api_result = self._convert_to_api_result(domain_result, start_time)

            logger.info(
                "Optimization completed with {}/{} successful trials; best objective={}",
                domain_result.successful_trials,
                domain_result.total_trials,
                domain_result.best_objective_value,
            )

            return api_result

        except Exception as e:
            logger.error("Optimization failed: {}", e)
            raise WorkflowError(
                f"Clean optimization failed: {e}",
                {"stage": "clean_optimization", "settings": str(settings)},
            ) from e

    def _convert_to_api_result(self, domain_result, start_time: float) -> APIOptimizationResult:
        """Convert domain OptimizationResult to API OptimizationResult.

        Args:
            domain_result: Domain optimization result
            start_time: Optimization start time

        Returns:
            API-compatible optimization result
        """
        # Create synthetic best trial for API compatibility
        best_trial_data = None
        if domain_result.best_trial:
            best_trial_data = _APICompatibleTrial(
                number=domain_result.best_trial.trial_number,
                value=domain_result.best_trial.objective_value or 0.0,
                params=domain_result.best_trial.hyperparameters,
            )

        # Calculate total duration
        total_duration = time.time() - start_time

        return APIOptimizationResult(
            best_trial=best_trial_data,
            training_result=domain_result.best_training_result,
            study_summary=domain_result.study_summary,
            duration_seconds=total_duration,
        )


class _APICompatibleTrial:
    """Trial wrapper for API compatibility.

    This provides both attribute and dictionary-style access
    to trial dataflow for backward compatibility.
    """

    def __init__(self, number: int, value: float, params: dict):
        """Initialize API-compatible trial.

        Args:
            number: Trial number
            value: Objective value
            params: Hyperparameters
        """
        self.number = number
        self.value = value
        self.params = params

    def __getattr__(self, name: str):
        """Provide attribute access."""
        if name in ("number", "value", "params"):
            return getattr(self, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __getitem__(self, key: str):
        """Provide dictionary-style access."""
        if key in ("number", "value", "params"):
            return getattr(self, key)
        raise KeyError(key)
