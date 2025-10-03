"""Data entry abstractions for FlexibleDataset configuration.

This module defines the core dataflow entry types used for configuring
flexible datasets. These are pure configuration objects with no
processing logic, following the single responsibility principle.
"""

from abc import ABC

import torch
from pydantic import Field, FilePath

from .core.base_settings import BasicSettings
from .transform_settings import TransformSettings
# Import moved to method level to avoid circular imports


class DataEntry(BasicSettings, ABC):
    """Base abstraction for dataflow configuration only.

    This class defines the common interface for all dataflow entries
    without any processing logic. Following the Open/Closed principle,
    it can be extended with new dataflow entry types without modification.

    Attributes:
        name: Unique identifier for this dataflow entry
        dtype: PyTorch dataflow type for the tensor dataflow
        required_in_loss: Whether this dataflow should be included in loss computation
        transforms: List of transforms to apply to this dataflow entry
    """

    name: str = Field(..., description="Unique identifier for this dataflow entry")
    dtype: torch.dtype | None = Field(
        default=None, description="PyTorch dataflow type. If None, uses session precision strategy."
    )
    required_in_loss: bool = Field(default=False, description="Include in loss computation")
    transforms: list[TransformSettings] = Field(
        default_factory=list, description="Transform chain for this dataflow entry"
    )

    def get_effective_dtype(self, precision_provider = None) -> torch.dtype:
        """Get the effective dtype for this dataflow entry.

        Args:
            precision_provider: Optional precision provider for strategy resolution.

        Returns:
            torch.dtype to use for this dataflow entry, resolved from explicit dtype
            or session precision strategy.
        """
        if self.dtype is not None:
            return self.dtype

        # Use precision service to resolve dtype from session/context
        from dlkit.interfaces.api.services.precision_service import get_precision_service

        precision_service = get_precision_service()
        return precision_service.get_torch_dtype(precision_provider)

    def resolve_dtype_with_fallback(
        self, fallback_dtype: torch.dtype = torch.float32
    ) -> torch.dtype:
        """Resolve dtype with explicit fallback.

        Args:
            fallback_dtype: Fallback dtype if no precision can be resolved.

        Returns:
            torch.dtype to use for this dataflow entry.
        """
        if self.dtype is not None:
            return self.dtype

        try:
            from dlkit.interfaces.api.services.precision_service import get_precision_service

            precision_service = get_precision_service()
            return precision_service.get_torch_dtype()
        except Exception:
            # Fallback to provided dtype if precision service fails
            return fallback_dtype


class Feature(DataEntry):
    """Input dataflow configuration for model features.

    Features represent input dataflow that will be fed to the model.
    They are file-based and typically not included in loss computation.

    Attributes:
        path: File path to the dataflow file
        required_in_loss: Defaults to False (features rarely used in loss)

    Notes:
        Autoencoder behaviour:
        - When the wrapper is configured with ``is_autoencoder = True`` and no explicit
          targets are provided in the dataset configuration, the processing pipeline
          (LossPairingStep) will automatically treat the feature entries as targets.
          This allows configuring autoencoders without duplicating the same entry under
          both features and targets.
    """

    path: FilePath = Field(..., description="Path to the feature dataflow file")
    required_in_loss: bool = Field(default=False, description="Features rarely used in loss")


class Target(DataEntry):
    """Expected output dataflow configuration.

    Targets represent the ground truth dataflow that the model should predict.
    They are file-based and typically included in loss computation.

    Attributes:
        path: File path to the target dataflow file
        write: Whether to save this target dataflow during inference
        required_in_loss: Defaults to True (targets typically used in loss)

    Notes:
        Strict pairing:
        - During training/validation/testing the processing pipeline enforces a strict
          1:1 mapping between target names and prediction names. If a target name does
          not have a corresponding prediction key, a clear error is raised listing the
          missing keys.
    """

    path: FilePath = Field(..., description="Path to the target dataflow file")
    write: bool = Field(
        default=False, description="Save the prediction related to this dataflow during inference"
    )
    required_in_loss: bool = Field(default=True, description="Targets typically used in loss")


class Latent(DataEntry):
    """Intermediate dataflow configuration (model-generated).

    Latents represent intermediate representations generated by the model
    during inference. They have no file path and are created dynamically.

    Note: Latents are NOT handled by the FlexibleDataset - they are managed
    by the processing pipeline during model inference.

    Attributes:
        write: Whether to save this latent dataflow during inference
        required_in_loss: Whether this latent should be included in loss
    """

    write: bool = Field(default=False, description="Save this latent during inference")
    required_in_loss: bool = Field(default=False, description="Latents rarely used in loss")


class Prediction(DataEntry):
    """Model prediction configuration.

    Predictions represent model outputs that correspond to specific targets.
    They are generated during model inference and can be compared to targets.

    Attributes:
        target_name: Name of the target this prediction corresponds to
        write: Whether to save prediction dataflow during inference
        required_in_loss: Defaults to True (predictions typically used in loss)
    """

    target_name: str = Field(..., description="Corresponding target name")
    write: bool = Field(default=True, description="Save predictions during inference")
    required_in_loss: bool = Field(default=True, description="Predictions typically used in loss")
