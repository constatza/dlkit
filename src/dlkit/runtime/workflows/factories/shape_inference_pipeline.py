"""Shape-inference helpers for build strategies."""

from __future__ import annotations

from dlkit.runtime.data.shape_inference import (
    infer_post_transform_shapes,
    infer_shapes_from_dataset,
)
from dlkit.shared.shapes import ShapeSummary
from dlkit.tools.config.data_entries import DataEntry


class ShapeInferencePipeline:
    """Centralize strategy-specific runtime shape inference."""

    def infer_flexible(
        self,
        model_name: object,
        dataset: object,
        selected_features: tuple[DataEntry, ...],
        selected_targets: tuple[DataEntry, ...],
    ) -> ShapeSummary | None:
        """Infer post-transform shapes for flexible builds."""
        try:
            return infer_post_transform_shapes(dataset, selected_features, selected_targets)
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Shape inference failed for '{model_name}'. "
                "Ensure dataset.__getitem__ returns a nested TensorDict with "
                "'features' and 'targets'. If transforms are applied, ensure all "
                "transforms have registered shape inference functions or specify "
                "explicit model init_kwargs (e.g., in_features=...)."
            ) from exc

    def infer_timeseries(self, dataset: object) -> ShapeSummary | None:
        """Infer shapes for timeseries datasets on a best-effort basis."""
        try:
            return infer_shapes_from_dataset(dataset)
        except ValueError, IndexError:
            return None
