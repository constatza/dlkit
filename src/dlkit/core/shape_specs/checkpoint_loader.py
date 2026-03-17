"""Utility for loading shape specifications from Lightning checkpoints.

This module provides a lightweight utility for extracting shape information
from checkpoints without loading the full model.
"""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import torch

from dlkit.tools.utils.logging_config import get_logger

from .core import IShapeSpec, create_shape_spec
from .serialization import VersionedShapeSerializer, SerializedShape

logger = get_logger(__name__)


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

        Note:
            Checkpoints must include ``dlkit_metadata``. Legacy checkpoints are
            no longer supported.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            # Load checkpoint with weights_only=False for custom classes
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}") from e

        if "dlkit_metadata" not in checkpoint:
            raise ValueError(
                "Checkpoint missing 'dlkit_metadata'. This checkpoint uses a legacy format "
                "that is no longer supported. Please re-train your model to generate "
                "a compatible checkpoint."
            )

        metadata = checkpoint["dlkit_metadata"]

        if "shape_spec" not in metadata:
            return None

        shape_data_dict = metadata["shape_spec"]

        try:
            # Deserialize shape spec
            serializer = VersionedShapeSerializer()

            # Deserialize shape data
            serialized = SerializedShape.from_dict(shape_data_dict)
            shape_data = serializer.deserialize(serialized)

            # Convert back to shape spec
            shapes = {name: entry.dimensions for name, entry in shape_data.entries.items()}
            return create_shape_spec(shapes)
        except Exception as e:
            logger.warning("Could not deserialize shape specification: {}", e)
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
        """Extract modern shape specification summary from checkpoint metadata.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Dictionary with shape information or error details
        """
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
