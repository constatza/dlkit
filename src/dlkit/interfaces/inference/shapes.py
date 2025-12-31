"""Shape inference logic for inference subsystem.

Consolidated shape inference with fallback strategies,
without hexagonal architecture overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dlkit.core.shape_specs import ShapeSpec
from dlkit.interfaces.api.domain.errors import WorkflowError


def infer_shape_specification(
    checkpoint: dict[str, Any],
    dataset: Any | None = None
) -> ShapeSpec:
    """Infer shape specification using fallback strategies.

    Strategy chain (in order):
    1. Try checkpoint metadata (fastest, most reliable)
    2. Fall back to dataset inference if provided
    3. Raise error if all strategies fail

    Args:
        checkpoint: Loaded checkpoint dictionary
        dataset: Optional dataset for shape inference fallback

    Returns:
        ShapeSpec: Inferred shape specification

    Raises:
        WorkflowError: If all inference strategies fail
    """
    # Strategy 1: Try checkpoint metadata first
    if "dlkit_metadata" in checkpoint and "shape_spec" in checkpoint["dlkit_metadata"]:
        try:
            shape_data = checkpoint["dlkit_metadata"]["shape_spec"]
            return ShapeSpec.from_dict(shape_data)
        except Exception as e:
            # Log but don't fail - try next strategy
            from loguru import logger
            logger.warning(f"Failed to load shape spec from checkpoint metadata: {e}")

    # Strategy 2: Fallback to dataset inference if provided
    if dataset is not None:
        from loguru import logger
        try:
            # TODO: Implement dataset-based shape inference
            # This requires importing from shape_specs which may have been refactored
            logger.warning("Dataset-based shape inference not yet implemented")
        except Exception as e:
            logger.error(f"Dataset shape inference failed: {e}")

    # All strategies failed
    raise WorkflowError(
        "Cannot infer shape: No valid shape source available. "
        "Checkpoint missing shape metadata and no dataset provided.",
        {"checkpoint_has_metadata": str("dlkit_metadata" in checkpoint)}
    )


def infer_shape_from_checkpoint_path(
    checkpoint_path: Path,
    dataset: Any | None = None
) -> ShapeSpec:
    """Convenience function to infer shapes from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file
        dataset: Optional dataset for fallback inference

    Returns:
        ShapeSpec: Inferred shape specification

    Raises:
        WorkflowError: If loading or inference fails
    """
    import torch

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        return infer_shape_specification(checkpoint, dataset)
    except Exception as e:
        raise WorkflowError(
            f"Failed to load checkpoint or infer shapes from {checkpoint_path}",
            {"error": str(e)}
        ) from e
