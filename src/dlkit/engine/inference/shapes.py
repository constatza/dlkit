"""Shape inference logic for inference subsystem.

Consolidated shape inference with fallback strategies,
without hexagonal architecture overhead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dlkit.common.errors import WorkflowError
from dlkit.common.shapes import ShapeSummary

if TYPE_CHECKING:
    pass


def infer_shape_specification(
    checkpoint: dict[str, Any], dataset: Any | None = None
) -> ShapeSummary | None:
    """Infer shape specification using fallback strategies.

    Strategy chain (in order):
    1. Try geometry from checkpoint metadata (preferred — new format)
    2. Try shape_summary from checkpoint metadata (legacy format)
    3. Fall back to dataset inference if provided
    4. Return None for external models that construct from kwargs

    Args:
        checkpoint: Loaded checkpoint dictionary
        dataset: Optional dataset for shape inference fallback

    Returns:
        ShapeSummary with in_shapes and out_shapes, or None if unavailable
    """
    from loguru import logger

    # Strategy 1: Try geometry from checkpoint metadata (new format)
    if "dlkit_metadata" in checkpoint:
        geometry_data = checkpoint["dlkit_metadata"].get("geometry", {})
        if geometry_data:
            try:
                shape_summary = _shape_summary_from_geometry(geometry_data)
                if shape_summary is not None:
                    return shape_summary
            except Exception as e:
                logger.warning(f"Failed to reconstruct ShapeSummary from geometry metadata: {e}")

        # Strategy 2: Legacy shape_summary format
        shape_data = checkpoint["dlkit_metadata"].get("shape_summary", {})
        in_shapes = shape_data.get("in_shapes") if shape_data else None
        out_shapes = shape_data.get("out_shapes") if shape_data else None
        if in_shapes and out_shapes:
            try:
                return ShapeSummary(
                    in_shapes=tuple(tuple(d) for d in in_shapes),
                    out_shapes=tuple(tuple(d) for d in out_shapes),
                )
            except Exception as e:
                logger.warning(f"Failed to reconstruct ShapeSummary from checkpoint metadata: {e}")

    # Strategy 3: Fallback to dataset inference if provided
    if dataset is not None:
        try:
            from dlkit.engine.data.shape_inference import infer_shapes_from_dataset

            return infer_shapes_from_dataset(dataset)
        except Exception as e:
            logger.error(f"Dataset shape inference failed: {e}")

    # No shape info available — external models construct from kwargs only
    return None


def _shape_summary_from_geometry(geometry_data: dict[str, Any]) -> ShapeSummary | None:
    """Reconstruct a ShapeSummary from serialized GeometrySpec data.

    Extracts FEATURE-role fields to derive in_shapes.  Output shapes are not
    stored in GeometrySpec (they belong to the contract), so out_shapes is not
    reconstructed here; callers must use the contract path when out_shapes are needed.

    Args:
        geometry_data: Dict produced by ``dataclasses.asdict(GeometrySpec)``.

    Returns:
        ShapeSummary if at least one FEATURE field is present, otherwise None.
    """
    from dlkit.common.geometry import FieldRole

    fields = geometry_data.get("fields", [])
    feature_fields = [f for f in fields if f.get("role") == FieldRole.FEATURE]
    if not feature_fields:
        return None
    in_shapes = tuple(tuple(int(d) for d in f["shape"]) for f in feature_fields)
    # out_shapes cannot be recovered from geometry alone; return a minimal summary
    # so the bridge can build a contract from in_shapes (it only needs out_shapes
    # for BranchTrunkSpec which requires 2+ in_shapes).
    return ShapeSummary(
        in_shapes=in_shapes,
        out_shapes=(),
    )


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
