"""Optuna-specific settings sampler implementation."""

from __future__ import annotations

from typing import cast

from dlkit.infrastructure.config.core.patching import patch_model
from dlkit.infrastructure.config.job_config import SearchJobConfig
from dlkit.infrastructure.config.search_settings import SearchSettings
from dlkit.infrastructure.utils.logging_config import get_logger

from .interfaces import ISettingsSampler, NullSettingsSampler, OptunaTrialProtocol

logger = get_logger(__name__)


class OptunaSettingsSampler:
    """Optuna implementation of settings sampling following SRP.

    Single Responsibility: Sample hyperparameters from search.space ranges
    and produce a complete SearchJobConfig ready for workflows.

    This class maintains strict separation of concerns - settings classes only
    hold configuration, this class performs sampling operations.
    """

    def __init__(self, search_settings: SearchSettings):
        """Initialize with search configuration.

        Args:
            search_settings: Search configuration containing the hyperparameter space
        """
        self._search_settings = search_settings
        self._validate_search_settings()

    def sample(self, trial: OptunaTrialProtocol, base_settings: SearchJobConfig) -> SearchJobConfig:
        """Sample hyperparameters from search.space and merge into base settings.

        Args:
            trial: Optuna trial object for suggestions
            base_settings: Base search job configuration with concrete default values

        Returns:
            Settings with sampled hyperparameters applied to model section

        Raises:
            ValueError: If search configuration is invalid
        """
        try:
            # Return unchanged if no search space defined
            if not self._search_settings.space:
                logger.debug("No search.space defined, returning base settings")
                return base_settings

            # Sample hyperparameters from search.space
            sampled_params = self._sample_model_parameters(trial)

            if not sampled_params:
                logger.debug("No parameters sampled, returning base settings")
                return base_settings

            # Apply sampled parameters to MODEL section
            return self._apply_sampled_parameters(base_settings, sampled_params)

        except Exception as e:
            logger.warning("Failed to sample hyperparameters: {}", e)
            return base_settings

    def _sample_model_parameters(
        self, trial: OptunaTrialProtocol
    ) -> dict[str, str | int | float | bool]:
        """Sample model parameters from search.space.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled parameter values with path keys
        """
        sampled_params = {}

        for param_path, space_param in self._search_settings.space.items():
            range_spec = space_param.model_dump()
            if self._is_range_specification(range_spec):
                try:
                    value = self._get_optuna_suggestion(trial, param_path, range_spec)
                    sampled_params[param_path] = value
                    logger.debug("Sampled parameter '{}'", param_path)
                except Exception as e:
                    logger.warning("Failed to sample '{}': {}", param_path, e)
            else:
                # Use constant value for non-range specs
                sampled_params[param_path] = space_param.model_dump().get("value", range_spec)
                logger.debug("Using concrete value for '{}'", param_path)

        return sampled_params

    def _apply_sampled_parameters(
        self,
        base_settings: SearchJobConfig,
        sampled_params: dict[str, str | int | float | bool],
    ) -> SearchJobConfig:
        """Apply sampled parameters to model section of base settings.

        Args:
            base_settings: Base search job configuration
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Updated settings with sampled parameters
        """
        if not getattr(base_settings, "model", None):
            logger.debug("No model in base settings, cannot apply sampled parameters")
            return base_settings

        try:
            # Patch model with sampled parameters using compile_mixed_overrides semantics
            updated_model = patch_model(base_settings.model, sampled_params)

            # Return new settings with updated model
            result = base_settings.model_copy(update={"model": updated_model})

            logger.debug("Applied {} sampled parameters to model", len(sampled_params))
            return result

        except Exception as e:
            logger.warning("Failed to apply sampled parameters: {}", e)
            return base_settings

    def _is_range_specification(
        self, spec: str | int | float | bool | list[str | int | float | bool] | dict
    ) -> bool:
        """Check if a specification defines a hyperparameter range.

        Args:
            spec: Parameter specification to check

        Returns:
            True if spec defines a range (has 'low'/'high' or 'choices')
        """
        return isinstance(spec, dict) and (("low" in spec and "high" in spec) or "choices" in spec)

    def _get_optuna_suggestion(
        self,
        trial: OptunaTrialProtocol,
        param_name: str,
        range_spec: dict[str, str | int | float | bool | list[str | int | float | bool]],
    ) -> str | int | float | bool:
        """Get Optuna suggestion for a parameter range.

        Replaces HyperParameterSettings.get_optuna_suggestion to maintain SRP.

        Args:
            trial: Optuna trial object
            param_name: Parameter name for suggestion
            range_spec: Range specification dictionary

        Returns:
            Suggested parameter value

        Raises:
            ValueError: If range specification is invalid
        """
        if not isinstance(range_spec, dict):
            raise ValueError(f"Invalid range specification for {param_name}: {range_spec}")

        param_type = range_spec.get("type")
        low = range_spec.get("low")
        high = range_spec.get("high")
        choices = range_spec.get("choices")

        # Handle categorical choices
        if choices is not None:
            if not isinstance(choices, (list, tuple)):
                raise ValueError(f"Choices must be list or tuple for {param_name}: {choices}")
            return trial.suggest_categorical(param_name, choices=list(choices))

        # Handle integer/float ranges
        if low is not None and high is not None:
            log_val = param_type in ("log_float", "log_int")
            step_val = range_spec.get("step")
            if param_type in ("int", "log_int"):
                return trial.suggest_int(
                    param_name,
                    low=cast(int, low),
                    high=cast(int, high),
                    step=cast(int, step_val) if step_val is not None else 1,
                    log=log_val,
                )
            return trial.suggest_float(
                param_name,
                low=cast(float, low),
                high=cast(float, high),
                step=cast(float, step_val) if step_val is not None else None,
                log=log_val,
            )

        raise ValueError(f"Invalid hyperparameter specification for {param_name}: {range_spec}")

    def _validate_search_settings(self) -> None:
        """Validate that search settings are properly configured.

        Raises:
            ValueError: If settings are invalid
        """
        if not self._search_settings:
            raise ValueError("SearchSettings cannot be None")

        if not hasattr(self._search_settings, "space"):
            raise ValueError("SearchSettings must have 'space' attribute")

        if not isinstance(self._search_settings.space, dict):
            raise ValueError("search.space must be a dictionary")


def create_settings_sampler(search_settings: SearchSettings | None = None) -> ISettingsSampler:
    """Factory function to create appropriate settings sampler.

    Args:
        search_settings: Search configuration; if None or empty space uses NullSettingsSampler.

    Returns:
        ISettingsSampler instance
    """
    if search_settings is None or not search_settings.space:
        return NullSettingsSampler()

    return OptunaSettingsSampler(search_settings)
