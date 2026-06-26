"""Model settings — model class selector and hyperparameters."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field, ModelWrapValidatorHandler, model_validator

from dlkit.common.types import ActivationName, NormalizerName
from dlkit.infrastructure.config.core.base_settings import BasicSettings


class ModelParams(BasicSettings):
    """Free-form model hyperparameters forwarded to the model __init__.

    Uses extra="allow" so any architecture-specific kwargs can be passed
    without modifying this class.

    Args:
        activation: Activation function name.
        normalize: Normalizer layer name.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    activation: ActivationName | str | None = None
    normalize: NormalizerName | None = None


class ModelSettings(BasicSettings):
    """Model class selector plus hyperparameters sub-table.

    Args:
        name: Model class name (alias: class).
        module_path: Python module path where the class is defined.
        checkpoint: Path to a pre-trained checkpoint file.
        params: Architecture-specific hyperparameters.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    name: str = Field(alias="class")
    module_path: str | None = None
    checkpoint: Path | None = None
    params: ModelParams = Field(default_factory=ModelParams)

    @model_validator(mode="wrap")
    @classmethod
    def _reject_duplicate_alias(
        cls, data: object, handler: ModelWrapValidatorHandler[ModelSettings]
    ) -> ModelSettings:
        """Reject input that specifies both alias and field name simultaneously.

        Pydantic 2 pre-processes alias resolution before ``mode="before"``
        validators receive the dict, so ``mode="wrap"`` is required to inspect
        the raw input before that normalisation occurs.

        Args:
            data: Raw input data before field validation.
            handler: Pydantic's continuation handler for normal validation.

        Returns:
            A validated ``ModelSettings`` instance.

        Raises:
            ValueError: If both 'name' and 'class' keys are present in the input.
        """
        if isinstance(data, dict) and "name" in data and "class" in data:
            raise ValueError(
                "Provide either 'class' (TOML alias) or 'name' (Python kwarg), not both."
            )
        return handler(data)
