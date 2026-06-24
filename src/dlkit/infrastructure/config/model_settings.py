"""Model settings — model class selector and hyperparameters."""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field

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
