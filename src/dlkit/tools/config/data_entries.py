"""Data entry abstractions for FlexibleDataset configuration.

This module defines the core dataflow entry types used for configuring
flexible datasets. These are pure configuration objects with no
processing logic, following the single responsibility principle.
"""

from abc import ABC
from pathlib import Path

import torch
import numpy as np
from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from .core.base_settings import BasicSettings
from .transform_settings import TransformSettings
# Import moved to method level to avoid circular imports


class DataEntry(BasicSettings, ABC):
    """Base abstraction for dataflow configuration only.

    This class defines the common interface for all dataflow entries
    without any processing logic. Following the Open/Closed principle,
    it can be extended with new dataflow entry types without modification.

    Design Philosophy:
    - Flexible data sources: Supports both file-based (path) and in-memory (value) data
    - Testability: Direct value assignment simplifies testing without file I/O
    - Backwards compatible: Existing path-based configs continue working

    Attributes:
        name: Unique identifier for this dataflow entry
        value: Optional in-memory tensor/array value (alternative to path)
        dtype: PyTorch dataflow type for the tensor dataflow
        required_in_loss: Whether this dataflow should be included in loss computation
        transforms: List of transforms to apply to this dataflow entry
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Unique identifier for this dataflow entry")
    value: torch.Tensor | np.ndarray | None = Field(
        default=None,
        description="In-memory tensor/array value (alternative to path for testing/programmatic use)",
        exclude=True,  # Exclude from serialization (too large)
    )
    dtype: torch.dtype | None = Field(
        default=None, description="PyTorch dataflow type. If None, uses session precision strategy."
    )
    required_in_loss: bool = Field(default=False, description="Include in loss computation")
    transforms: list[TransformSettings] = Field(
        default_factory=list, description="Transform chain for this dataflow entry"
    )

    def has_value(self) -> bool:
        """Check if this entry has an in-memory value.

        Returns:
            True if value is present, False otherwise
        """
        return self.value is not None

    def has_path(self) -> bool:
        """Check if this entry has a file path.

        Returns:
            True if path attribute exists and is set, False otherwise
        """
        return hasattr(self, "path") and self.path is not None

    def get_effective_dtype(self, precision_provider=None) -> torch.dtype:
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
    Data can be provided via file path OR in-memory value.

    Attributes:
        path: Optional file path to the dataflow file (XOR with value)
        value: Optional in-memory tensor/array (XOR with path, inherited from DataEntry)
        required_in_loss: Defaults to False (features rarely used in loss)

    Validation:
        Exactly one of `path` or `value` must be provided (not both, not neither).

    Notes:
        Autoencoder behaviour:
        - When the wrapper is configured with ``is_autoencoder = True`` and no explicit
          targets are provided in the dataset configuration, the processing pipeline
          (LossPairingStep) will automatically treat the feature entries as targets.
          This allows configuring autoencoders without duplicating the same entry under
          both features and targets.
    """

    path: Path | None = Field(
        default=None, description="Path to the feature dataflow file (XOR with value)"
    )
    required_in_loss: bool = Field(default=False, description="Features rarely used in loss")

    @model_validator(mode="after")
    def validate_path_or_value(self) -> "Feature":
        """Validate that exactly one of path or value is provided.

        Returns:
            The validated Feature instance

        Raises:
            ValueError: If neither or both path and value are provided
        """
        has_value = self.has_value()
        has_path = self.has_path()

        if not (has_value or has_path):
            raise ValueError(
                f"Feature '{self.name}' must have either 'path' or 'value' specified. "
                f"Config files should specify 'path = \"path/to/file\"'. "
                f"For programmatic/testing use, specify 'value = <array>'."
            )
        if has_path and has_value:
            raise ValueError(
                f"Feature '{self.name}' cannot have both 'path' and 'value' specified (use one or the other)"
            )

        return self


class Target(DataEntry):
    """Expected output dataflow configuration.

    Targets represent the ground truth dataflow that the model should predict.
    Data can be provided via file path OR in-memory value.

    Attributes:
        path: Optional file path to the target dataflow file (XOR with value)
        value: Optional in-memory tensor/array (XOR with path, inherited from DataEntry)
        write: Whether to save this target dataflow during inference
        required_in_loss: Defaults to True (targets typically used in loss)

    Validation:
        Exactly one of `path` or `value` must be provided (not both, not neither).

    Notes:
        Strict pairing:
        - During training/validation/testing the processing pipeline enforces a strict
          1:1 mapping between target names and prediction names. If a target name does
          not have a corresponding prediction key, a clear error is raised listing the
          missing keys.
    """

    path: Path | None = Field(
        default=None, description="Path to the target dataflow file (XOR with value)"
    )
    write: bool = Field(
        default=False, description="Save the prediction related to this dataflow during inference"
    )
    required_in_loss: bool = Field(default=True, description="Targets typically used in loss")

    @model_validator(mode="after")
    def validate_path_or_value(self) -> "Target":
        """Validate that exactly one of path or value is provided.

        Returns:
            The validated Target instance

        Raises:
            ValueError: If neither or both path and value are provided
        """
        has_value = self.has_value()
        has_path = self.has_path()

        if not (has_value or has_path):
            raise ValueError(
                f"Target '{self.name}' must have either 'path' or 'value' specified. "
                f"Config files should specify 'path = \"path/to/file\"'. "
                f"For programmatic/testing use, specify 'value = <array>'."
            )
        if has_path and has_value:
            raise ValueError(
                f"Target '{self.name}' cannot have both 'path' and 'value' specified (use one or the other)"
            )

        return self


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


class AutoencoderTarget(Target):
    """Autoencoder reconstruction target that references a feature entry.

    AutoencoderTarget provides automatic transform inversion for autoencoder architectures.
    It references a Feature entry and automatically creates an inverted transform chain
    that applies inverse_transform() in reverse order.

    This is essential for transforms like SampleNormL2 that store per-sample state
    during forward() and require the same state for inverse_transform().

    Attributes:
        feature_ref: Name of the Feature entry to reference
        path: Copied from referenced feature (do not specify manually)
        value: Copied from referenced feature (do not specify manually)
        transforms: Derived from feature (do not specify manually - will be ignored)
        write: Whether to save reconstruction dataflow during inference
        required_in_loss: Defaults to True (reconstructions used in loss)

    Validation:
        - feature_ref must be specified
        - Referenced feature must exist in entry_configs
        - Referenced entry must be a Feature instance
        - Transforms should not be manually specified (derived from feature)

    Example:
        ```python
        x = Feature(
            name="x",
            path="input.npy",
            transforms=[
                TransformSettings(name="StandardScaler"),
                TransformSettings(name="SampleNormL2"),
            ],
        )

        y = AutoencoderTarget(
            name="y",
            feature_ref="x",  # Automatically inverts x's transforms
        )

        # Runtime: x transforms applied forward, y transforms applied inverse
        # SampleNormL2._last_norms preserved from forward to inverse
        ```
    """

    feature_ref: str = Field(
        ..., description="Name of the Feature entry to reference for transform inversion"
    )

    # Override parent validators since path/value will be copied from feature_ref
    @model_validator(mode="after")
    def validate_feature_ref(self) -> "AutoencoderTarget":
        """Validate feature_ref and warn about manual transforms.

        Returns:
            The validated AutoencoderTarget instance

        Raises:
            ValueError: If feature_ref is not provided
        """
        if not self.feature_ref:
            raise ValueError("AutoencoderTarget must have 'feature_ref' specified")

        # Warn if transforms are manually specified (will be overridden)
        if self.transforms:
            import warnings

            warnings.warn(
                f"AutoencoderTarget '{self.name}' has transforms specified, but these "
                f"will be derived from feature_ref '{self.feature_ref}'. "
                f"Manual transforms specification will be ignored.",
                UserWarning,
            )

        # Note: feature_ref existence validation happens at runtime in wrapper
        # because we don't have access to entry_configs during model validation

        return self

    def has_path(self) -> bool:
        """Check if this entry has a file path.

        For AutoencoderTarget, path comes from referenced feature.

        Returns:
            True if path attribute exists and is set, False otherwise
        """
        # Path will be set at runtime from feature_ref
        return hasattr(self, "path") and self.path is not None

    def has_value(self) -> bool:
        """Check if this entry has an in-memory value.

        For AutoencoderTarget, value comes from referenced feature.

        Returns:
            True if value is present, False otherwise
        """
        # Value will be set at runtime from feature_ref
        return self.value is not None


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
