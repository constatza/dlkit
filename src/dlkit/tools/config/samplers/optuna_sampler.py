"""Optuna-specific settings sampler implementation."""

from __future__ import annotations

from typing import Any

from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.core.patching import patch_model
from dlkit.tools.config.optuna_settings import OptunaSettings
from dlkit.tools.utils.logging_config import get_logger

from .interfaces import ISettingsSampler

logger = get_logger(__name__)


class OptunaSettingsSampler:
    """Optuna implementation of settings sampling following SRP.

    Single Responsibility: Sample hyperparameters from OPTUNA.model ranges
    and produce complete GeneralSettings ready for workflows.

    This class obsoletes HyperParameterSettings.sample() method to maintain
    strict separation of concerns - settings classes only hold configuration,
    this class performs sampling operations.
    """

    def __init__(self, optuna_settings: OptunaSettings):
        """Initialize with Optuna configuration.

        Args:
            optuna_settings: OPTUNA configuration containing model ranges
        """
        self._optuna_settings = optuna_settings
        self._validate_optuna_settings()

    def sample(self, trial: Any, base_settings: GeneralSettings) -> GeneralSettings:
        """Sample hyperparameters from OPTUNA.model and merge into base settings.

        Args:
            trial: Optuna trial object for suggestions
            base_settings: Base configuration with concrete default values

        Returns:
            GeneralSettings with sampled hyperparameters applied to MODEL section

        Raises:
            ValueError: If OPTUNA configuration is invalid
        """
        try:
            # Return unchanged if no model ranges defined
            if not self._optuna_settings.has_model_ranges:
                logger.debug("No OPTUNA.model ranges defined, returning base settings")
                return base_settings

            # Sample hyperparameters from OPTUNA.model ranges
            sampled_params = self._sample_model_parameters(trial)

            if not sampled_params:
                logger.debug("No parameters sampled, returning base settings")
                return base_settings

            # Apply sampled parameters to MODEL section
            return self._apply_sampled_parameters(base_settings, sampled_params)

        except Exception as e:
            logger.warning("Failed to sample hyperparameters: {}", e)
            return base_settings

    def _sample_model_parameters(self, trial: Any) -> dict[str, Any]:
        """Sample model parameters from OPTUNA.model ranges.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of sampled parameter values with path keys
        """
        sampled_params = {}

        for param_path, range_spec in self._optuna_settings.model.items():
            if self._is_range_specification(range_spec):
                try:
                    value = self._get_optuna_suggestion(trial, param_path, range_spec)
                    sampled_params[param_path] = value
                    logger.debug("Sampled parameter '{}'", param_path)
                except Exception as e:
                    logger.warning("Failed to sample '{}': {}", param_path, e)
            else:
                # Use concrete value if not a range specification
                sampled_params[param_path] = range_spec
                logger.debug("Using concrete value for '{}'", param_path)

        return sampled_params

    def _apply_sampled_parameters(
        self, base_settings: GeneralSettings, sampled_params: dict[str, Any]
    ) -> GeneralSettings:
        """Apply sampled parameters to MODEL section of base settings.

        Args:
            base_settings: Base configuration settings
            sampled_params: Dictionary of sampled parameter values

        Returns:
            Updated GeneralSettings with sampled parameters
        """
        if not base_settings.MODEL:
            logger.debug("No MODEL in base settings, cannot apply sampled parameters")
            return base_settings

        try:
            # Patch MODEL with sampled parameters using compile_mixed_overrides semantics
            updated_model = patch_model(base_settings.MODEL, sampled_params)

            # Return new settings with updated MODEL
            result = base_settings.patch({"MODEL": updated_model})

            logger.debug("Applied {} sampled parameters to MODEL", len(sampled_params))
            return result

        except Exception as e:
            logger.warning("Failed to apply sampled parameters: {}", e)
            return base_settings

    def _is_range_specification(self, spec: Any) -> bool:
        """Check if a specification defines a hyperparameter range.

        Args:
            spec: Parameter specification to check

        Returns:
            True if spec defines a range (has 'low'/'high' or 'choices')
        """
        return isinstance(spec, dict) and (("low" in spec and "high" in spec) or "choices" in spec)

    def _get_optuna_suggestion(
        self, trial: Any, param_name: str, range_spec: dict[str, Any]
    ) -> Any:
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

        low = range_spec.get("low")
        high = range_spec.get("high")
        choices = range_spec.get("choices")

        if low and high and choices:
            raise ValueError(
                f"You have defined mutually exclusive hyperparameter choices for {param_name}: "
                f"`low`, `high` are incompatible with `choices`"
            )

        # Handle integer/float ranges
        if low and high:
            step = range_spec.get("step", 1)
            log = range_spec.get("log", False)
            # Determine if this is an integer or float range
            if all(isinstance(val, int) for val in [low, high, step]):
                return trial.suggest_int(param_name, low=low, high=high, step=step, log=log)
            return trial.suggest_float(param_name, low=low, high=high, step=step, log=log)

        # Handle categorical choices
        if choices:
            if not isinstance(choices, (list, tuple)):
                raise ValueError(f"Choices must be list or tuple for {param_name}: {choices}")
            return trial.suggest_categorical(param_name, choices=list(choices))

        raise ValueError(f"Invalid hyperparameter specification for {param_name}: {range_spec}")

    def _validate_optuna_settings(self) -> None:
        """Validate that OPTUNA settings are properly configured.

        Raises:
            ValueError: If settings are invalid
        """
        if not self._optuna_settings:
            raise ValueError("OptunaSettings cannot be None")

        if not hasattr(self._optuna_settings, "model"):
            raise ValueError("OPTUNA settings must have 'model' attribute")

        if not isinstance(self._optuna_settings.model, dict):
            raise ValueError("OPTUNA.model must be a dictionary")


def create_settings_sampler(optuna_settings: OptunaSettings | None = None) -> ISettingsSampler:
    """Factory function to create appropriate settings sampler.

    Args:
        optuna_settings: OPTUNA configuration, if None uses NullSettingsSampler

    Returns:
        ISettingsSampler instance
    """
    from .interfaces import NullSettingsSampler

    if optuna_settings is None or not optuna_settings.enabled:
        return NullSettingsSampler()

    return OptunaSettingsSampler(optuna_settings)
