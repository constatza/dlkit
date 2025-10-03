"""Standalone transform chain executor for inference.

This module provides transform execution capabilities that operate independently
from Lightning, preserving the exact same transform functionality while
enabling minimal-overhead inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dlkit.core.training.transforms.chain import TransformChain
from .checkpoint_loader import CheckpointTransformLoader


class TransformChainExecutor:
    """Standalone executor for transform chains in inference.

    This class provides transform application capabilities that work independently
    from Lightning wrappers, enabling direct transform usage in production
    inference scenarios.

    The executor maintains the exact same transform behavior as during training
    while operating with minimal overhead for production use.
    """

    def __init__(
        self,
        fitted_transforms: dict[str, TransformChain] | None = None,
        checkpoint_path: Path | str | None = None
    ) -> None:
        """Initialize the transform chain executor.

        Args:
            fitted_transforms: Pre-loaded fitted transform chains
            checkpoint_path: Path to checkpoint containing fitted transforms

        Note:
            Either fitted_transforms or checkpoint_path must be provided.
            If both are provided, fitted_transforms takes precedence.
        """
        self._fitted_transforms: dict[str, TransformChain] = {}

        if fitted_transforms is not None:
            self._fitted_transforms = fitted_transforms.copy()
        elif checkpoint_path is not None:
            self._load_transforms_from_checkpoint(checkpoint_path)

    def _load_transforms_from_checkpoint(self, checkpoint_path: Path | str) -> None:
        """Load fitted transforms from checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        loader = CheckpointTransformLoader()
        self._fitted_transforms = loader.load_fitted_transforms(checkpoint_path)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | str) -> TransformChainExecutor:
        """Create executor by loading transforms from checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            TransformChainExecutor: Configured executor with loaded transforms

        Example:
            >>> executor = TransformChainExecutor.from_checkpoint("model.ckpt")
            >>> transformed_inputs = executor.apply_feature_transforms(raw_inputs)
        """
        return cls(checkpoint_path=checkpoint_path)

    @classmethod
    def from_transforms(cls, fitted_transforms: dict[str, TransformChain]) -> TransformChainExecutor:
        """Create executor with pre-loaded transform chains.

        Args:
            fitted_transforms: Dictionary of fitted transform chains

        Returns:
            TransformChainExecutor: Configured executor
        """
        return cls(fitted_transforms=fitted_transforms)

    def has_transforms(self) -> bool:
        """Check if any transform chains are available.

        Returns:
            True if transform chains are available for application
        """
        return len(self._fitted_transforms) > 0

    def get_available_transforms(self) -> list[str]:
        """Get list of available transform chain names.

        Returns:
            List of transform chain names
        """
        return list(self._fitted_transforms.keys())

    def has_transform_for(self, name: str) -> bool:
        """Check if a transform chain is available for a specific entry.

        Args:
            name: Entry name to check

        Returns:
            True if transform chain is available for the entry
        """
        return name in self._fitted_transforms

    def apply_feature_transforms(
        self,
        inputs: dict[str, Tensor],
        feature_names: list[str] | None = None
    ) -> dict[str, Tensor]:
        """Apply fitted feature transforms to input tensors.

        Args:
            inputs: Dictionary of input tensors
            feature_names: List of feature names to transform (None = transform all available)

        Returns:
            Dictionary of transformed tensors

        Example:
            >>> inputs = {"x": torch.randn(32, 10), "y": torch.randn(32, 5)}
            >>> transformed = executor.apply_feature_transforms(inputs, ["x"])
            >>> # Only "x" is transformed, "y" is unchanged
        """
        if not self.has_transforms():
            return inputs.copy()

        transformed_inputs = {}
        names_to_transform = feature_names if feature_names is not None else list(inputs.keys())

        for name, tensor in inputs.items():
            if name in names_to_transform and self.has_transform_for(name):
                try:
                    transform_chain = self._fitted_transforms[name]
                    transformed_inputs[name] = transform_chain(tensor)
                except Exception as e:
                    # Log warning and use original tensor
                    print(f"Warning: Failed to apply transform to '{name}': {e}")
                    transformed_inputs[name] = tensor
            else:
                # No transform available or not requested - use original
                transformed_inputs[name] = tensor

        return transformed_inputs

    def apply_inverse_target_transforms(
        self,
        outputs: dict[str, Tensor],
        target_names: list[str] | None = None
    ) -> dict[str, Tensor]:
        """Apply inverse transforms to model outputs.

        Args:
            outputs: Dictionary of model output tensors
            target_names: List of target names to inverse transform (None = transform all available)

        Returns:
            Dictionary of inverse-transformed tensors

        Example:
            >>> outputs = {"y": torch.randn(32, 5), "prediction": torch.randn(32, 1)}
            >>> original_scale = executor.apply_inverse_target_transforms(outputs, ["y"])
            >>> # "y" is inverse-transformed back to original scale
        """
        if not self.has_transforms():
            return outputs.copy()

        inverse_transformed = {}
        names_to_transform = target_names if target_names is not None else list(outputs.keys())

        for name, tensor in outputs.items():
            if name in names_to_transform and self.has_transform_for(name):
                try:
                    transform_chain = self._fitted_transforms[name]
                    if hasattr(transform_chain, 'inverse_transform'):
                        inverse_transformed[name] = transform_chain.inverse_transform(tensor)
                    else:
                        # No inverse available - use original
                        inverse_transformed[name] = tensor
                except Exception as e:
                    # Log warning and use original tensor
                    print(f"Warning: Failed to apply inverse transform to '{name}': {e}")
                    inverse_transformed[name] = tensor
            else:
                # No transform available or not requested - use original
                inverse_transformed[name] = tensor

        return inverse_transformed

    def transform_single_input(self, name: str, tensor: Tensor) -> Tensor:
        """Apply transform to a single named input tensor.

        Args:
            name: Name of the input entry
            tensor: Input tensor to transform

        Returns:
            Transformed tensor (original if no transform available)
        """
        if self.has_transform_for(name):
            try:
                transform_chain = self._fitted_transforms[name]
                return transform_chain(tensor)
            except Exception as e:
                print(f"Warning: Failed to apply transform to '{name}': {e}")

        return tensor

    def inverse_transform_single_output(self, name: str, tensor: Tensor) -> Tensor:
        """Apply inverse transform to a single named output tensor.

        Args:
            name: Name of the output entry
            tensor: Output tensor to inverse transform

        Returns:
            Inverse-transformed tensor (original if no inverse available)
        """
        if self.has_transform_for(name):
            try:
                transform_chain = self._fitted_transforms[name]
                if hasattr(transform_chain, 'inverse_transform'):
                    return transform_chain.inverse_transform(tensor)
            except Exception as e:
                print(f"Warning: Failed to apply inverse transform to '{name}': {e}")

        return tensor

    def get_transform_info(self) -> dict[str, dict[str, Any]]:
        """Get information about available transform chains.

        Returns:
            Dictionary with transform chain information

        Example:
            >>> info = executor.get_transform_info()
            >>> print(info["x"]["has_inverse"])
            True
        """
        info = {}

        for name, transform_chain in self._fitted_transforms.items():
            info[name] = {
                "has_inverse": hasattr(transform_chain, 'inverse_transform'),
                "num_transforms": len(transform_chain.transforms) if hasattr(transform_chain, 'transforms') else 0,
                "is_fitted": getattr(transform_chain, 'is_fitted', True),
            }

        return info