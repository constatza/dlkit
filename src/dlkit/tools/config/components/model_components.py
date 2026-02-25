"""Model component settings without build() methods - pure configuration."""

from typing import TYPE_CHECKING
from collections.abc import Callable
from pathlib import Path

from lightning import LightningModule
from pydantic import Field, FilePath, field_validator
from pydantic_settings import SettingsConfigDict
from torchmetrics import Metric
import torch.nn as nn

from dlkit.core.datatypes.base import IntHyperparameter
from ..core.base_settings import ComponentSettings, HyperParameterSettings, BasicSettings
from ..optimizer_settings import OptimizerSettings, SchedulerSettings

if TYPE_CHECKING:
    pass


def _validate_batch_key(v: str) -> str:
    """Validate 'namespace.entry_name' key strings.

    Args:
        v: Key string to validate.

    Returns:
        Validated key string.

    Raises:
        ValueError: If key format is invalid.
    """
    parts = v.split(".", 1)
    if len(parts) != 2 or parts[0] not in ("features", "targets"):
        raise ValueError(
            f"key must be 'features.<entry_name>' or 'targets.<entry_name>', got '{v}'"
        )
    return v


class LossInputRef(BasicSettings, frozen=True):
    """Maps a loss function kwarg to a TensorDict key in the batch.

    Attributes:
        arg: Kwarg name in the loss function, e.g. "matrix"
        key: Batch key in "namespace.entry_name" format, e.g. "features.A"
    """

    arg: str = Field(..., description="Kwarg name in the loss function")
    key: str = Field(..., description="Batch key in 'namespace.entry_name' format")

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        return _validate_batch_key(v)


class MetricInputRef(BasicSettings, frozen=True):
    """Maps a metric function kwarg to a TensorDict key in the batch.

    Attributes:
        arg: Kwarg name in the metric function
        key: Batch key in "namespace.entry_name" format
    """

    arg: str = Field(..., description="Kwarg name in the metric function")
    key: str = Field(..., description="Batch key in 'namespace.entry_name' format")

    @field_validator("key")
    @classmethod
    def _validate_key(cls, v: str) -> str:
        return _validate_batch_key(v)


class MetricComponentSettings(ComponentSettings[Metric]):
    """Configuration for metrics components - pure configuration only.

    This replaces MetricSettings.build() with factory pattern.

    Args:
        component_name: Name/class of the metric
        module_path: Module path to the metric
        target_key: Batch key for metric target in 'namespace.entry_name' format
        extra_inputs: Extra kwargs passed to the metric, routed from batch
    """

    name: str = Field(default="MeanSquaredError", description="Name of the metric")
    module_path: str = Field(
        default="torchmetrics.regression", description="Module path to the metric"
    )
    target_key: str | None = Field(
        default=None,
        description="Batch key for metric target in 'namespace.entry_name' format. None = first targets/ entry in config.",
    )
    extra_inputs: tuple[MetricInputRef, ...] = Field(
        default=(),
        description="Extra kwargs passed to the metric, routed from batch.",
    )

    @field_validator("target_key")
    @classmethod
    def _validate_target_key(cls, v: str | None) -> str | None:
        if v is not None:
            _validate_batch_key(v)
        return v


class LossComponentSettings(ComponentSettings[Callable]):
    """Configuration for loss function components - pure configuration only.

    This replaces LossSettings.build() with factory pattern.

    The default module path uses dlkit's shared functional module, which provides:
    - Standard losses (mse, mae, etc.) from torchmetrics.functional
    - Custom differentiable losses (huber, quantile, normalized, etc.)
    - All functions are pure, differentiable, and usable as both loss and metric

    Args:
        component_name: Name/class of the loss function
        module_path: Module path to the loss function
        target_key: Batch key for loss target in 'namespace.entry_name' format
        extra_inputs: Extra kwargs passed to the loss function, routed from batch

    Examples:
        >>> # Standard MSE loss (default)
        >>> LossComponentSettings()  # Uses mse from dlkit.core.training.functional
        >>>
        >>> # Custom loss from shared module
        >>> LossComponentSettings(name="huber_loss")
        >>>
        >>> # External loss from torchmetrics
        >>> LossComponentSettings(
        ...     name="mean_absolute_error",
        ...     module_path="torchmetrics.functional.regression"
        ... )
    """

    name: str | Callable = Field(
        default="mse", description="Name of the loss function (default: mse)"
    )
    module_path: str = Field(
        default="dlkit.core.training.functional",
        description="Module path to the loss function (default: shared functional module)",
    )
    target_key: str | None = Field(
        default=None,
        description="Batch key for loss target in 'namespace.entry_name' format. None = first targets/ entry in config.",
    )
    extra_inputs: tuple[LossInputRef, ...] = Field(
        default=(),
        description="Extra kwargs passed to the loss function, routed from batch.",
    )

    @field_validator("target_key")
    @classmethod
    def _validate_target_key(cls, v: str | None) -> str | None:
        if v is not None:
            _validate_batch_key(v)
        return v


class ModelComponentSettings(ComponentSettings[LightningModule], HyperParameterSettings):
    """Model architecture configuration - pure configuration only.

    This replaces ModelSettings.build() with factory pattern.
    All model construction logic is moved to factories.

    Checkpoint Usage (Workflow-Specific):
    - Training resume: Use TRAINING.resume_from_checkpoint (full training state)
    - Inference: Use checkpoint field below (model weights only)
    - Standalone inference: Use InferenceConfig.model_checkpoint_path

    The checkpoint field in this class is for inference workflows only.
    For resuming training, use TRAINING.resume_from_checkpoint instead.

    Args:
        component_name: Model class name or type
        module_path: Module path to the model
        checkpoint: Checkpoint path for inference (model weights only, NOT for training resume)
        shape: Input shape for the model
        Various model architecture parameters as hyperparameters
    """

    model_config = SettingsConfigDict(extra="allow", arbitrary_types_allowed=True)

    name: str | type[nn.Module] = Field(..., description="Model namespace path")
    module_path: str = Field(default="dlkit.core.models.nn", description="Module path to the model")

    # Checkpoint for inference (NOT for training resume)
    checkpoint: Path | str | None = Field(
        default=None,
        description=(
            "Checkpoint path for inference workflows (model weights only). "
            "For resuming training, use TRAINING.resume_from_checkpoint instead."
        ),
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
    def has_metrics(self) -> bool:
        """Check if metrics are configured.

        Returns:
            bool: True if metrics are specified
        """
        return len(self.metrics) > 0
