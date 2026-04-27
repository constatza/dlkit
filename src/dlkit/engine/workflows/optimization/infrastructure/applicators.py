"""Concrete hyperparameter applicator implementations.

These applicators implement the IHyperparameterApplicator protocol
to apply sampled hyperparameters to workflow settings.
"""

from typing import Any


class ModelSettingsApplicator:
    """Applies sampled hyperparameters to the MODEL section of settings.

    This applicator patches the MODEL configuration with hyperparameters
    when they are present and the base settings have a MODEL section.
    """

    def apply(
        self,
        base_settings: Any,
        hyperparameters: dict[str, Any],
    ) -> Any:
        """Apply hyperparameters to MODEL settings.

        Args:
            base_settings: Base workflow settings
            hyperparameters: Sampled hyperparameters to apply

        Returns:
            Settings with MODEL hyperparameters applied
        """
        if base_settings.MODEL and hyperparameters:
            updated_model = base_settings.MODEL.patch(hyperparameters)
            return base_settings.patch({"MODEL": updated_model})
        return base_settings
