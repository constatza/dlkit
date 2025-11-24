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

    def load_fitted_transforms(
        self,
        checkpoint_path: Path | str
    ) -> tuple[dict[str, TransformChain], dict[str, TransformChain]]:
        """Load fitted transform chains from a checkpoint file.

        Separates Feature and Target transforms from the beginning for clear
        separation of concerns and fail-fast ambiguity detection.

        Args:
            checkpoint_path: Path to the Lightning checkpoint file

        Returns:
            Tuple of (feature_transforms, target_transforms) dictionaries

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint doesn't contain fitted transforms
            ValueError: If transform loading fails

        Example:
            >>> loader = CheckpointTransformLoader()
            >>> feature_transforms, target_transforms = loader.load_fitted_transforms("model.ckpt")
            >>> x_transform = feature_transforms.get("x")
            >>> y_transform = target_transforms.get("y")
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load checkpoint with weights_only=False for DLKit custom classes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        # Extract fitted transforms from the checkpoint (separated by type)
        feature_transforms, target_transforms = self._extract_fitted_transforms(checkpoint)

        return feature_transforms, target_transforms

    def _extract_fitted_transforms(
        self,
        checkpoint: dict[str, Any]
    ) -> tuple[dict[str, TransformChain], dict[str, TransformChain]]:
        """Extract fitted transforms from checkpoint data, separated by type.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            Tuple of (feature_transforms, target_transforms) dictionaries

        Raises:
            ValueError: If transform extraction fails
        """
        from dlkit.tools.config.data_entries import Feature, Target

        feature_transforms = {}
        target_transforms = {}

        state_dict = checkpoint.get("state_dict", {})
        inference_metadata = checkpoint.get("inference_metadata", {})
        entry_configs = inference_metadata.get("entry_configs", {})

        # Guard: No entry configs means we can't separate transforms
        if not entry_configs:
            return feature_transforms, target_transforms

        # Check for modern format (separate feature/target) or legacy format
        has_modern = self._has_modern_format(state_dict)
        has_legacy = self._has_legacy_format(state_dict)

        # Extract based on format
        if has_modern:
            feature_transforms = self._load_transforms_by_prefix("fitted_feature_transforms", state_dict, entry_configs)
            target_transforms = self._load_transforms_by_prefix("fitted_target_transforms", state_dict, entry_configs)
        elif has_legacy:
            # Legacy format - separate during loading
            all_transforms = self._load_transforms_by_prefix("fitted_transforms", state_dict, entry_configs)
            feature_transforms, target_transforms = self._separate_by_type(all_transforms, entry_configs)

        return feature_transforms, target_transforms

    def _has_modern_format(self, state_dict: dict[str, Any]) -> bool:
        """Check if checkpoint uses modern format (separate feature/target transforms)."""
        return any(k.startswith("fitted_feature_transforms.") or k.startswith("fitted_target_transforms.")
                  for k in state_dict.keys())

    def _has_legacy_format(self, state_dict: dict[str, Any]) -> bool:
        """Check if checkpoint uses legacy format (mixed fitted_transforms)."""
        return any(k.startswith("fitted_transforms.") for k in state_dict.keys())

    def _load_transforms_by_prefix(
        self,
        prefix: str,
        state_dict: dict[str, Any],
        entry_configs: dict[str, Any]
    ) -> dict[str, TransformChain]:
        """Load all transforms with given prefix from state_dict."""
        transforms = {}

        # Group state dict keys by entry name
        transform_state_dicts = self._group_by_entry_name(prefix, state_dict)

        # Reconstruct each TransformChain
        for entry_name, transform_state in transform_state_dicts.items():
            chain = self._reconstruct_chain(entry_name, transform_state, entry_configs)
            if chain is not None:
                transforms[entry_name] = chain

        return transforms

    def _group_by_entry_name(self, prefix: str, state_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Group state dict keys by entry name."""
        grouped = {}

        for key in state_dict.keys():
            if not key.startswith(f"{prefix}."):
                continue

            # Parse: "prefix.entry_name.param.path" -> "entry_name"
            parts = key.split(".", 2)
            if len(parts) < 2:
                continue

            entry_name = parts[1]
            param_path = parts[2] if len(parts) > 2 else ""

            if entry_name not in grouped:
                grouped[entry_name] = {}

            if param_path:
                grouped[entry_name][param_path] = state_dict[key]

        return grouped

    def _reconstruct_chain(
        self,
        entry_name: str,
        transform_state: dict[str, Any],
        entry_configs: dict[str, Any]
    ) -> TransformChain | None:
        """Reconstruct a TransformChain from state dict."""
        try:
            # Get transform settings from entry config
            entry_config = entry_configs.get(entry_name)
            transform_settings = []

            if entry_config and hasattr(entry_config, 'transforms'):
                transform_settings = entry_config.transforms
            elif entry_config and isinstance(entry_config, dict):
                transform_settings = entry_config.get('transforms', [])

            # Create and load chain
            chain = TransformChain(transform_settings)
            result = chain.load_state_dict(transform_state, strict=False)

            # Register unexpected keys as buffers
            self._register_unexpected_buffers(chain, result.unexpected_keys, transform_state)

            return chain

        except Exception as e:
            print(f"Warning: Failed to load transform chain for '{entry_name}': {e}")
            return None

    def _register_unexpected_buffers(
        self,
        chain: TransformChain,
        unexpected_keys: list[str],
        transform_state: dict[str, Any]
    ) -> None:
        """Register unexpected keys as buffers in the chain."""
        if not unexpected_keys:
            return

        for unexpected_key in unexpected_keys:
            try:
                parts = unexpected_key.split('.')
                module = chain
                for part in parts[:-1]:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                buffer_name = parts[-1]
                buffer_value = transform_state[unexpected_key]
                module.register_buffer(buffer_name, buffer_value)
            except Exception:
                pass  # Silently skip registration failures

    def _separate_by_type(
        self,
        all_transforms: dict[str, TransformChain],
        entry_configs: dict[str, Any]
    ) -> tuple[dict[str, TransformChain], dict[str, TransformChain]]:
        """Separate transforms into feature and target based on entry_configs."""
        from dlkit.tools.config.data_entries import Feature, Target

        feature_transforms = {}
        target_transforms = {}

        for name, chain in all_transforms.items():
            entry_config = entry_configs.get(name)
            if isinstance(entry_config, Target):
                target_transforms[name] = chain
            elif isinstance(entry_config, Feature):
                feature_transforms[name] = chain

        return feature_transforms, target_transforms

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