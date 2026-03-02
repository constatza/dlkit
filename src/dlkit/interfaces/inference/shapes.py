"""Shape inference logic for inference subsystem.

Consolidated shape inference with fallback strategies,
without hexagonal architecture overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlkit.interfaces.api.domain.errors import WorkflowError

if TYPE_CHECKING:
    from dlkit.core.shape_specs.simple_inference import ShapeSummary


def infer_shape_specification(
    checkpoint: dict[str, Any], dataset: Any | None = None
) -> ShapeSummary | None:
    """Infer shape specification using fallback strategies.

    Strategy chain (in order):
    1. Try checkpoint metadata (fastest, most reliable)
    2. Fall back to dataset inference if provided
    3. Return None for external models that construct from kwargs

    Args:
        checkpoint: Loaded checkpoint dictionary
        dataset: Optional dataset for shape inference fallback

    Returns:
        ShapeSummary with in_shapes and out_shapes, or None if unavailable
    """
    from loguru import logger
    from dlkit.core.shape_specs.simple_inference import ShapeSummary

    # Strategy 1: Try checkpoint metadata first
    if "dlkit_metadata" in checkpoint and "shape_summary" in checkpoint["dlkit_metadata"]:
        shape_data = checkpoint["dlkit_metadata"]["shape_summary"]
        in_shapes = shape_data.get("in_shapes")
        out_shapes = shape_data.get("out_shapes")
        if in_shapes and out_shapes:
            try:
                return ShapeSummary(
                    in_shapes=tuple(tuple(d) for d in in_shapes),
                    out_shapes=tuple(tuple(d) for d in out_shapes),
                )
            except Exception as e:
                logger.warning(f"Failed to reconstruct ShapeSummary from checkpoint metadata: {e}")

    # Strategy 2: Fallback to dataset inference if provided
    if dataset is not None:
        try:
            from dlkit.core.shape_specs.simple_inference import infer_shapes_from_dataset

            return infer_shapes_from_dataset(dataset)
        except Exception as e:
            logger.error(f"Dataset shape inference failed: {e}")

    # No shape info available — external models construct from kwargs only
    return None


def infer_shape_from_checkpoint_path(checkpoint_path: Path, dataset: Any | None = None) -> Any:
    """Convenience function to infer shapes from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file
        dataset: Optional dataset for fallback inference

    Returns:
        ShapeSummary or similar shape data object

    Raises:
        WorkflowError: If loading or inference fails
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return infer_shape_specification(checkpoint, dataset)
    except Exception as e:
        raise WorkflowError(
            f"Failed to load checkpoint or infer shapes from {checkpoint_path}", {"error": str(e)}
        ) from e
