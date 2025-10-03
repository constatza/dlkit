"""Inference configuration.

This module defines configuration classes for inference that operate
independently from training configurations, requiring only the information
available in checkpoints and user inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field, ConfigDict, field_validator

# from dlkit.core.datatypes.dataset import Shape  # Removed - using IShapeSpec


class InferenceConfig(BaseModel):
    """Configuration for inference.

    This configuration contains only the essential information needed for
    standalone inference, independent of training datasets or configurations.
    Everything required is extracted from the model checkpoint.

    Attributes:
        model_checkpoint_path: Path to the trained model checkpoint
        feature_names: Names of input features (e.g., ["x", "input", "features"])
        target_names: Names of expected outputs (e.g., ["y", "output", "predictions"])
        transform_names: Names of available fitted transforms
        model_shape: Input shape information for the model
        device: Device to run inference on ("auto", "cpu", "cuda", "mps")
        dtype: Default tensor dtype for inputs
        batch_size: Default batch size for processing
        apply_transforms: Whether to apply fitted transforms to inputs/outputs
    """

    model_checkpoint_path: Path = Field(
        ..., description="Path to the trained model checkpoint"
    )

    feature_names: list[str] = Field(
        default_factory=list,
        description="Names of input features expected by the model"
    )

    target_names: list[str] = Field(
        default_factory=list,
        description="Names of target outputs produced by the model"
    )

    transform_names: list[str] | None = Field(
        default=None,
        description="Names of available fitted transform chains"
    )

    model_shape: dict[str, tuple[int, ...]] | None = Field(
        default=None,
        description="Input shape information for the model (mapping feature names to shapes)"
    )

    device: str = Field(
        default="auto",
        description="Device to run inference on (auto/cpu/cuda/mps)"
    )

    dtype: torch.dtype = Field(
        default=torch.float32,
        description="Default tensor dtype for inputs"
    )

    batch_size: int = Field(
        default=32,
        description="Default batch size for processing"
    )

    apply_transforms: bool = Field(
        default=True,
        description="Whether to apply fitted transforms to inputs/outputs"
    )

    # Additional metadata from checkpoint
    wrapper_settings: dict[str, Any] = Field(
        default_factory=dict,
        description="Original wrapper settings from training"
    )

    entry_configs: dict[str, Any] = Field(
        default_factory=dict,
        description="Original data entry configurations"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('model_shape')
    @classmethod
    def validate_model_shape(cls, v):
        """Convert shape values to tuples for consistency."""
        if v is None:
            return v

        if isinstance(v, dict):
            converted = {}
            for key, value in v.items():
                if isinstance(value, (list, tuple)):
                    converted[key] = tuple(int(x) for x in value)
                else:
                    converted[key] = (int(value),)
            return converted

        return v

    def resolve_device(self) -> torch.device:
        """Resolve the target device for inference.

        Returns:
            torch.device: Resolved device for inference
        """
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(self.device)

    def get_feature_names(self) -> list[str]:
        """Get list of feature names expected by the model.

        Returns:
            List of feature names for input validation
        """
        return self.feature_names.copy()

    def get_target_names(self) -> list[str]:
        """Get list of target names produced by the model.

        Returns:
            List of target names for output processing
        """
        return self.target_names.copy()

    def has_transforms(self) -> bool:
        """Check if fitted transforms are available.

        Returns:
            True if transforms are available for application
        """
        return (
            self.transform_names is not None and
            len(self.transform_names) > 0 and
            self.apply_transforms
        )

    def validate_inputs(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Validate input structure against expected features.

        Args:
            inputs: Dictionary of input tensors/arrays

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        if not self.feature_names:
            # No validation if feature names not specified
            return errors

        provided_features = set(inputs.keys())
        expected_features = set(self.feature_names)

        missing_features = expected_features - provided_features
        if missing_features:
            errors["missing_features"] = f"Missing required features: {list(missing_features)}"

        extra_features = provided_features - expected_features
        if extra_features:
            errors["extra_features"] = f"Unexpected features provided: {list(extra_features)}"

        return errors


# Rebuild the model to properly handle the Shape type reference
InferenceConfig.model_rebuild()