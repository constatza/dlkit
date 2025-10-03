"""Model component settings without build() methods - pure configuration."""

from typing import TYPE_CHECKING
from collections.abc import Callable

from lightning import LightningModule
from pydantic import Field, FilePath
from pydantic_settings import SettingsConfigDict
from torchmetrics import Metric
import torch.nn as nn

from dlkit.core.datatypes.base import IntHyperparameter
from ..core.base_settings import ComponentSettings, HyperParameterSettings
from ..optimizer_settings import OptimizerSettings, SchedulerSettings

if TYPE_CHECKING:
    pass


class MetricComponentSettings(ComponentSettings[Metric]):
    """Configuration for metrics components - pure configuration only.

    This replaces MetricSettings.build() with factory pattern.

    Args:
        component_name: Name/class of the metric
        module_path: Module path to the metric
    """

    name: str = Field(default="MeanSquaredError", description="Name of the metric")
    module_path: str = Field(
        default="torchmetrics.regression", description="Module path to the metric"
    )


class LossComponentSettings(ComponentSettings[Callable]):
    """Configuration for loss function components - pure configuration only.

    This replaces LossSettings.build() with factory pattern.

    Args:
        component_name: Name/class of the loss function
        module_path: Module path to the loss function
    """

    name: str | Callable = Field(
        default="mean_squared_error", description="Name of the loss function"
    )
    module_path: str = Field(
        default="torchmetrics.functional.regression", description="Module path to the loss function"
    )


class ModelComponentSettings(ComponentSettings[LightningModule], HyperParameterSettings):
    """Model architecture configuration - pure configuration only.

    This replaces ModelSettings.build() with factory pattern.
    All model construction logic is moved to factories.

    The checkpoint field is optional for training (resume training) but required for inference.
    This allows the same config file to work for both modes.

    Args:
        component_name: Model class name or type
        module_path: Module path to the model
        checkpoint: Optional checkpoint path (required for inference, optional for training)
        shape: Input shape for the model
        Various model architecture parameters as hyperparameters
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str | type[nn.Module] = Field(..., description="Model namespace path")
    module_path: str = Field(default="dlkit.core.models.nn", description="Module path to the model")

    # Shared checkpoint field - optional for training, required for inference
    checkpoint: FilePath | None = Field(
        default=None,
        description="Path to model checkpoint (optional for training, required for inference)",
    )


    # Model architecture hyperparameters
    heads: IntHyperparameter | None = Field(default=None, description="Number of heads")
    num_layers: IntHyperparameter | None = Field(default=None, description="Number of layers")
    hidden_size: IntHyperparameter | None = Field(default=None, description="Hidden dimension size")
    latent_size: IntHyperparameter | None = Field(default=None, description="Latent dimension size")
    kernel_size: IntHyperparameter | None = Field(
        default=None, description="Convolution kernel size"
    )
    latent_channels: IntHyperparameter | None = Field(
        default=None, description="Number of latent channels before reduce to vector"
    )
    latent_width: IntHyperparameter | None = Field(
        default=None, description="Latent width before reduce to vector"
    )
    latent_height: IntHyperparameter | None = Field(
        default=None, description="Latent height before reduce to vector"
    )
    in_channels: IntHyperparameter | None = Field(
        default=None, description="Number of input channels"
    )
    out_channels: IntHyperparameter | None = Field(
        default=None, description="Number of output channels"
    )
    num_heads: IntHyperparameter | None = Field(default=None, description="Number of heads")



class WrapperComponentSettings(ComponentSettings[nn.Module]):
    """Model wrapper configuration - pure configuration only.

    This replaces WrapperSettings.build() with factory pattern.
    Contains all training-related model configuration.

    Note:
        The `feature_transforms` and `target_transforms` fields are deprecated.
        Transforms should be specified on individual dataflow entries (Feature/Target)
        and are applied by the processing pipeline (TransformApplicationStep).

    Args:
        component_name: Wrapper class name
        module_path: Module path to wrapper
        optimizer: Optimizer configuration
        scheduler: Scheduler configuration
        checkpoint: Model checkpoint path
        Training flags and loss configuration
        Transform configurations
        Metrics configuration
    """

    name: str | type[nn.Module] = Field(
        default="StandardLightningWrapper", description="Name of the wrapper"
    )
    module_path: str = Field(
        default="dlkit.core.models.wrappers", description="Module path to the wrapper"
    )

    # Training components
    optimizer: OptimizerSettings = Field(
        default_factory=OptimizerSettings, description="Optimizer settings"
    )
    scheduler: SchedulerSettings = Field(
        default_factory=SchedulerSettings, description="Scheduler settings"
    )

    # Model state
    checkpoint: FilePath | None = Field(default=None, description="Path to model checkpoint")

    # Training flags
    train: bool = Field(default=True, description="Whether to train the model")
    test: bool = Field(default=True, description="Whether to test the model")
    predict: bool = Field(default=True, description="Whether to predict with the model")

    # Loss function and metrics for training
    loss_function: LossComponentSettings = Field(
        default_factory=LossComponentSettings,
        description="Loss function settings for training and validation",
    )
    metrics: tuple[MetricComponentSettings, ...] = Field(
        default=tuple(), description="List of metrics to compute on the model at test time"
    )

    # Model type flags
    is_autoencoder: bool = Field(
        default=False,
        description=(
            "Whether the model is an autoencoder. When True and the dataset has no explicit "
            "targets, the processing pipeline will pair each feature with itself for loss "
            "computation (via LossPairingStep)."
        ),
    )

    @property
    def has_checkpoint(self) -> bool:
        """Check if checkpoint path is configured.

        Returns:
            bool: True if checkpoint path is set
        """
        return self.checkpoint is not None

    @property
    def has_metrics(self) -> bool:
        """Check if metrics are configured.

        Returns:
            bool: True if metrics are specified
        """
        return len(self.metrics) > 0
