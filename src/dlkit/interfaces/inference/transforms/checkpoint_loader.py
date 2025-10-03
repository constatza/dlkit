"""Checkpoint transform loading for inference.

This module provides utilities to load fitted transform chains directly
from Lightning checkpoints for standalone inference execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import ModuleDict

from dlkit.core.training.transforms.chain import TransformChain
from dlkit.core.shape_specs import IShapeSpec, CheckpointShapeLoader


class CheckpointTransformLoader:
    """Loader for extracting fitted transform chains from checkpoints.

    This class handles the loading and extraction of fitted transform chains
    that were saved during training, making them available for standalone
    inference.
    """

    def __init__(self) -> None:
        """Initialize the checkpoint transform loader."""
        pass

    def load_fitted_transforms(self, checkpoint_path: Path | str) -> dict[str, TransformChain]:
        """Load fitted transform chains from a checkpoint file.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            Dictionary mapping entry names to fitted transform chains

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint doesn't contain fitted transforms
            ValueError: If transform loading fails

        Example:
            >>> loader = CheckpointTransformLoader()
            >>> transforms = loader.load_fitted_transforms("model.ckpt")
            >>> x_transform = transforms.get("x")
            >>> if x_transform:
            ...     transformed_x = x_transform(raw_x)
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load checkpoint with weights_only=False for DLKit custom classes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        # Extract fitted transforms from the checkpoint
        fitted_transforms = self._extract_fitted_transforms(checkpoint)

        if not fitted_transforms:
            # Return empty dict if no transforms available (e.g., graph/timeseries models)
            return {}

        return fitted_transforms

    def _extract_fitted_transforms(self, checkpoint: dict[str, Any]) -> dict[str, TransformChain]:
        """Extract fitted transforms from checkpoint data.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            Dictionary of fitted transform chains

        Raises:
            ValueError: If transform extraction fails
        """
        fitted_transforms = {}

        # Check for fitted_transforms in the state_dict (Lightning ModuleDict)
        state_dict = checkpoint.get("state_dict", {})

        # Look for fitted_transforms keys in state_dict
        transform_keys = [key for key in state_dict.keys() if key.startswith("fitted_transforms.")]

        if transform_keys:
            # Reconstruct ModuleDict from state_dict
            module_dict = ModuleDict()

            # Extract transform state dicts
            transform_state_dicts = {}
            for key in transform_keys:
                # Parse key: "fitted_transforms.entry_name.layer.weight" -> "entry_name"
                parts = key.split(".", 2)
                if len(parts) >= 2:
                    entry_name = parts[1]
                    param_path = ".".join(parts[2:]) if len(parts) > 2 else ""

                    if entry_name not in transform_state_dicts:
                        transform_state_dicts[entry_name] = {}

                    if param_path:
                        transform_state_dicts[entry_name][param_path] = state_dict[key]

            # Reconstruct TransformChain instances
            for entry_name, transform_state_dict in transform_state_dicts.items():
                try:
                    # Create empty TransformChain and load state
                    # Use a placeholder shape that will be updated when state is loaded
                    placeholder_shape = (1, 1)  # Minimal placeholder shape
                    transform_chain = TransformChain([], input_shape=placeholder_shape)
                    transform_chain.load_state_dict(transform_state_dict)
                    fitted_transforms[entry_name] = transform_chain
                except Exception as e:
                    # Log warning but continue with other transforms
                    print(f"Warning: Failed to load transform chain for '{entry_name}': {e}")

        return fitted_transforms

    def get_inference_metadata(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Get inference metadata from checkpoint.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            Inference metadata dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint doesn't contain inference metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        if "inference_metadata" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain inference metadata. "
                "This checkpoint may be from an older version of DLKit."
            )

        return checkpoint["inference_metadata"]

    def has_transforms(self, checkpoint_path: Path | str) -> bool:
        """Check if checkpoint contains fitted transforms.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            True if fitted transforms are available
        """
        try:
            transforms = self.load_fitted_transforms(checkpoint_path)
            return len(transforms) > 0
        except Exception:
            return False

    def get_transform_names(self, checkpoint_path: Path | str) -> list[str]:
        """Get names of available transform chains in checkpoint.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            List of transform chain names
        """
        try:
            transforms = self.load_fitted_transforms(checkpoint_path)
            return list(transforms.keys())
        except Exception:
            return []

    def validate_checkpoint_compatibility(self, checkpoint_path: Path | str) -> dict[str, str]:
        """Validate checkpoint compatibility for inference.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        try:
            checkpoint_path = Path(checkpoint_path)

            if not checkpoint_path.exists():
                errors["file_not_found"] = f"Checkpoint file not found: {checkpoint_path}"
                return errors

            # Try to load checkpoint
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as e:
                errors["load_failed"] = f"Failed to load checkpoint: {e}"
                return errors

            # Check for required components
            if "state_dict" not in checkpoint:
                errors["missing_state_dict"] = "Checkpoint missing model state_dict"

            if "inference_metadata" not in checkpoint:
                errors["missing_metadata"] = (
                    "Checkpoint missing inference metadata. "
                    "This checkpoint may not support inference."
                )

            # Check metadata structure
            if "inference_metadata" in checkpoint:
                metadata = checkpoint["inference_metadata"]
                required_fields = ["feature_names", "target_names"]
                for field in required_fields:
                    if field not in metadata:
                        errors[f"missing_{field}"] = f"Inference metadata missing {field}"

        except Exception as e:
            errors["validation_failed"] = f"Checkpoint validation failed: {e}"

        return errors

    def load_shape_spec(self, checkpoint_path: Path | str) -> IShapeSpec | None:
        """Load shape specification from checkpoint.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            Shape specification if available, None otherwise

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint cannot be loaded
        """
        return CheckpointShapeLoader.load_shape_spec(checkpoint_path)

    def has_shape_spec(self, checkpoint_path: Path | str) -> bool:
        """Check if checkpoint contains shape specification.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            True if checkpoint contains shape specification
        """
        return CheckpointShapeLoader.has_shape_metadata(checkpoint_path)

    def create_model_from_checkpoint(
        self,
        checkpoint_path: Path | str,
        model_class: type | None = None
    ) -> Any:
        """Create model instance from checkpoint with automatic shape loading.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file
            model_class: Optional model class to instantiate (if None, uses checkpoint metadata)

        Returns:
            Model instance with shape specification loaded from checkpoint

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint cannot be loaded or model cannot be created
        """
        import torch
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Load checkpoint to extract metadata
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Load shape spec from checkpoint
        shape_spec = self.load_shape_spec(checkpoint_path)

        if shape_spec is None:
            raise ValueError(
                f"No shape specification found in checkpoint {checkpoint_path}. "
                "Cannot create model without shape information."
            )

        # Try to extract model settings from checkpoint for complete reconstruction
        model_settings = None
        if isinstance(checkpoint, dict) and 'dlkit_metadata' in checkpoint:
            metadata = checkpoint['dlkit_metadata']
            if 'model_settings' in metadata:
                model_settings = metadata['model_settings']

        # If model_class is provided, create instance with available parameters
        if model_class is not None:
            from dlkit.core.models.nn.base import ShapeAwareModel, ShapeAgnosticModel

            if issubclass(model_class, ShapeAwareModel):
                # Try to use model settings from checkpoint if available
                if model_settings and 'params' in model_settings:
                    try:
                        params = model_settings['params']
                        return model_class(unified_shape=shape_spec, **params)
                    except TypeError:
                        # Fallback if params don't match - just use shape
                        return model_class(unified_shape=shape_spec)
                else:
                    # No params available - just use shape (may fail for some models)
                    return model_class(unified_shape=shape_spec)
            elif issubclass(model_class, ShapeAgnosticModel):
                # Create shape-agnostic model (doesn't need shape)
                if model_settings and 'params' in model_settings:
                    try:
                        params = model_settings['params']
                        return model_class(**params)
                    except TypeError:
                        return model_class()
                else:
                    return model_class()
            else:
                # Legacy model - try with shape_spec parameter
                try:
                    return model_class(shape_spec=shape_spec)
                except TypeError:
                    # Fallback without shape
                    return model_class()

        return None