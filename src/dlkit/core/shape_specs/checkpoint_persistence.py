"""Shape checkpoint persistence for seamless inference.

This module provides automatic saving and loading of shape specifications
in PyTorch Lightning checkpoints, enabling seamless inference without
re-specifying shape information.
"""

from __future__ import annotations

from typing import Any, Dict
import json
from pathlib import Path

import torch
from lightning.pytorch import LightningModule

from .core import IShapeSpec, ShapeSpec, NullShapeSpec
from .value_objects import ShapeData, ModelFamily, ShapeSource
from .serialization import VersionedShapeSerializer, SerializationFormat


class ShapeCheckpointMixin:
    """Mixin to add shape persistence to Lightning modules.

    This mixin automatically saves shape specifications during checkpointing
    and provides utilities to load them back during inference.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the mixin."""
        super().__init__(*args, **kwargs)
        self._saved_shape_spec: IShapeSpec | None = None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save shape specification in checkpoint.

        Args:
            checkpoint: Checkpoint dictionary to modify
        """
        super().on_save_checkpoint(checkpoint)  # type: ignore

        # Extract shape spec from model if it's shape-aware
        shape_spec = self._extract_shape_spec()

        if shape_spec is not None and not shape_spec.is_empty():
            # Serialize shape spec for checkpoint
            serializer = VersionedShapeSerializer()

            # Extract shape data from spec
            from .value_objects import ShapeData, ShapeEntry, ModelFamily, ShapeSource

            entries = {}
            all_shapes = shape_spec.get_all_shapes()
            for name, dimensions in all_shapes.items():
                entries[name] = ShapeEntry(name=name, dimensions=dimensions)

            shape_data = ShapeData(
                entries=entries,
                model_family=ModelFamily.DLKIT_NN,  # Default
                source=ShapeSource.TRAINING_DATASET
            )

            serialized_shape = serializer.serialize(shape_data)

            # Store in checkpoint metadata
            if "shape_metadata" not in checkpoint:
                checkpoint["shape_metadata"] = {}

            checkpoint["shape_metadata"]["shape_spec"] = serialized_shape.to_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load shape specification from checkpoint.

        Args:
            checkpoint: Checkpoint dictionary
        """
        super().on_load_checkpoint(checkpoint)  # type: ignore

        # Load shape spec if available
        if "shape_metadata" in checkpoint and "shape_spec" in checkpoint["shape_metadata"]:
            try:
                serializer = VersionedShapeSerializer()
                shape_data_dict = checkpoint["shape_metadata"]["shape_spec"]

                # Deserialize shape data
                from .serialization import SerializedShape
                serialized = SerializedShape.from_dict(shape_data_dict)
                shape_data = serializer.deserialize(serialized)

                # Convert back to shape spec
                from .core import create_shape_spec
                shapes = {name: entry.dimensions for name, entry in shape_data.entries.items()}
                self._saved_shape_spec = create_shape_spec(shapes)
            except Exception as e:
                # Log warning but don't fail the load
                print(f"Warning: Could not load shape specification from checkpoint: {e}")
                self._saved_shape_spec = None

    def _extract_shape_spec(self) -> IShapeSpec | None:
        """Extract shape specification from the model.

        Returns:
            Shape specification if available, None otherwise
        """
        # Check if we have a wrapped model with shape spec
        if hasattr(self, 'model'):
            model = getattr(self, 'model')

            # Check for ABC-based models
            from ..models.nn.base import ShapeAwareModel
            if isinstance(model, ShapeAwareModel):
                return model.get_unified_shape()

            # Check for legacy shape spec access
            if hasattr(model, 'get_shape_spec'):
                return model.get_shape_spec()

        # Check if this module itself has shape spec
        if hasattr(self, '_unified_shape'):
            return getattr(self, '_unified_shape')

        if hasattr(self, '_shape_spec'):
            return getattr(self, '_shape_spec')

        return None

    def get_checkpoint_shape_spec(self) -> IShapeSpec | None:
        """Get the shape specification loaded from checkpoint.

        Returns:
            Shape specification from checkpoint, or None if not available
        """
        return self._saved_shape_spec


class CheckpointShapeLoader:
    """Utility for loading shape specifications from Lightning checkpoints.

    This class provides static methods for extracting shape information
    from checkpoints without loading the full model.
    """

    @staticmethod
    def load_shape_spec(checkpoint_path: Path | str) -> IShapeSpec | None:
        """Load shape specification from checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Shape specification if available, None otherwise

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint cannot be loaded
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load checkpoint with weights_only=False for custom classes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        # Extract shape metadata from new format: dlkit_metadata -> shape_spec
        if "dlkit_metadata" not in checkpoint or "shape_spec" not in checkpoint["dlkit_metadata"]:
            return None

        shape_data_dict = checkpoint["dlkit_metadata"]["shape_spec"]

        try:
            # Deserialize shape spec
            serializer = VersionedShapeSerializer()

            # Deserialize shape data
            from .serialization import SerializedShape
            serialized = SerializedShape.from_dict(shape_data_dict)
            shape_data = serializer.deserialize(serialized)

            # Convert back to shape spec
            from .core import create_shape_spec
            shapes = {name: entry.dimensions for name, entry in shape_data.entries.items()}
            return create_shape_spec(shapes)
        except Exception as e:
            print(f"Warning: Could not deserialize shape specification: {e}")
            return None

    @staticmethod
    def has_shape_metadata(checkpoint_path: Path | str) -> bool:
        """Check if checkpoint contains shape metadata.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            True if checkpoint contains shape metadata
        """
        try:
            shape_spec = CheckpointShapeLoader.load_shape_spec(checkpoint_path)
            return shape_spec is not None and not shape_spec.is_empty()
        except Exception:
            return False

    @staticmethod
    def extract_shape_info(checkpoint_path: Path | str) -> Dict[str, Any]:
        """Extract modern shape specification summary from checkpoint metadata."""
        try:
            shape_spec = CheckpointShapeLoader.load_shape_spec(checkpoint_path)

            if shape_spec is None or shape_spec.is_empty():
                return {
                    "has_shapes": False,
                    "details": "No shape_spec metadata present",
                }

            return {
                "has_shapes": True,
                "model_family": shape_spec.model_family(),
                "input_shape": shape_spec.get_input_shape(),
                "output_shape": shape_spec.get_output_shape(),
                "all_shapes": shape_spec.get_all_shapes(),
            }
        except Exception as exc:
            return {
                "has_shapes": False,
                "details": f"Error extracting shape metadata: {exc}",
            }


def enable_shape_persistence(module: LightningModule) -> LightningModule:
    """Enable shape persistence for a Lightning module.

    This function dynamically adds the ShapeCheckpointMixin to an existing
    Lightning module to enable automatic shape persistence.

    Args:
        module: Lightning module to enhance

    Returns:
        Enhanced module with shape persistence capabilities
    """
    # Create a new class that inherits from both the original class and the mixin
    class ShapeAwareModule(ShapeCheckpointMixin, module.__class__):
        pass

    # Copy the instance to the new class
    module.__class__ = ShapeAwareModule

    # Initialize mixin state
    module._saved_shape_spec = None

    return module
