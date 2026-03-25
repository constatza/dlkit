"""Simple shape inference from TensorDict dataset samples.

This module provides the ShapeSummary dataclass and infer_shapes_from_dataset
pure function that replaces the complex IShapeSpec subsystem for model
construction purposes.

Phase 2 (Post-Transform Shapes): Propagates dataset shapes through transform
chains to compute post-transform shapes for model construction.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeSummary:
    """Minimal shape info extracted from one dataset sample.

    Args:
        in_shapes: Shapes of feature tensors, one per Feature entry in config order.
        out_shapes: Shapes of target tensors, one per Target entry in config order.
    """

    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]

    @property
    def in_features(self) -> int:
        """Primary input feature size (first dim of first feature tensor).

        Returns:
            First dimension of the first feature tensor.
        """
        return self.in_shapes[0][0]

    @property
    def out_features(self) -> int:
        """Primary output feature size (first dim of first target tensor).

        Returns:
            First dimension of the first target tensor.
        """
        return self.out_shapes[0][0]

    @property
    def in_channels(self) -> int:
        """Input channels for conv models (alias for first dim of first feature).

        Returns:
            First dimension of the first feature tensor.
        """
        return self.in_shapes[0][0]

    @property
    def in_length(self) -> int:
        """Sequence length for conv/timeseries models (second dim of first feature).

        Returns:
            Second dimension of the first feature tensor.
        """
        return self.in_shapes[0][1]


def infer_shapes_from_dataset(dataset: Any) -> ShapeSummary:
    """Sample index 0 from dataset and extract shapes from a nested TensorDict sample.

    Args:
        dataset: Any dataset object whose __getitem__ returns a nested TensorDict.

    Returns:
        ShapeSummary with in_shapes and out_shapes extracted from the sample.

    Raises:
        ValueError: If dataset[0] does not return a nested TensorDict sample.
    """
    sample = dataset[0]

    try:
        from tensordict import TensorDictBase

        if isinstance(sample, TensorDictBase):
            feat_td = cast(TensorDictBase, sample["features"])
            targ_td = cast(TensorDictBase, sample["targets"])
            in_shapes = tuple(tuple(int(d) for d in feat_td[k].shape) for k in feat_td.keys())
            out_shapes = tuple(tuple(int(d) for d in targ_td[k].shape) for k in targ_td.keys())
            return ShapeSummary(in_shapes=in_shapes, out_shapes=out_shapes)
    except ImportError:
        pass

    raise ValueError(
        f"Expected dataset[0] to return a nested TensorDict with 'features' and 'targets', "
        f"got {type(sample).__name__}. Update your dataset's __getitem__ accordingly."
    )


def _resolve_transform_class(transform_settings: Any) -> type:
    """Resolve the transform class from its settings.

    Args:
        transform_settings: A TransformSettings instance or similar with name and
            optional module_path attributes.

    Returns:
        The resolved transform class.

    Raises:
        AttributeError: If transform_settings has no name attribute.
        ImportError: If module cannot be imported.
        AttributeError: If transform class cannot be found in module.
    """
    name = getattr(transform_settings, "name", None)
    raw_path = getattr(transform_settings, "module_path", None)
    module_path: str = raw_path or "dlkit.core.training.transforms"

    # If name is already a type, return it directly
    if isinstance(name, type):
        return name

    if not isinstance(name, str):
        raise TypeError(f"Expected transform name to be a str or type, got {type(name).__name__}")

    import importlib

    module = importlib.import_module(module_path)
    return cast(type, getattr(module, name))


def _extract_transform_kwargs(transform_settings: Any) -> dict[str, Any]:
    """Extract shape-inference kwargs from transform settings.

    Excludes name and module_path fields, extracting only parameters needed
    for shape inference (e.g., n_components, keep, dims).

    Args:
        transform_settings: A TransformSettings instance or similar.

    Returns:
        Dictionary of shape-inference kwargs.
    """
    exclude = {"name", "module_path"}
    with suppress(AttributeError):
        return {k: v for k, v in transform_settings.model_dump().items() if k not in exclude}
    return {}


def _propagate_shape_through_chain(
    shape: tuple[int, ...],
    transform_settings_list: Sequence[Any],
) -> tuple[int, ...]:
    """Propagate a single shape through a transform chain.

    Uses the SHAPE_INFERENCE_REGISTRY to analytically compute output shapes
    without executing transforms or instantiating models.

    Args:
        shape: Input shape as a tuple of ints.
        transform_settings_list: Sequence of TransformSettings instances.

    Returns:
        Output shape after all transforms have been applied.

    Raises:
        ValueError: If a transform has no registered shape inference function,
            preventing analytical shape propagation.
    """
    from dlkit.core.training.transforms.shape_inference import SHAPE_INFERENCE_REGISTRY

    current = shape
    for ts in transform_settings_list:
        cls = _resolve_transform_class(ts)
        if cls not in SHAPE_INFERENCE_REGISTRY:
            raise ValueError(
                f"Transform '{getattr(ts, 'name', cls)}' has no registered shape inference "
                "function. Cannot determine post-transform shapes analytically. "
                "Specify explicit model init_kwargs (e.g., in_features=...) to bypass shape inference."
            )
        current = SHAPE_INFERENCE_REGISTRY[cls](current, **_extract_transform_kwargs(ts))
    return current


def _propagate_entry_shapes(
    raw_shapes: tuple[tuple[int, ...], ...],
    entries: Sequence[Any],
) -> tuple[tuple[int, ...], ...]:
    """Propagate shapes through transform chains for a sequence of entries.

    For each entry, applies its configured transforms to the corresponding
    raw shape. Falls back to raw_shapes if entries count doesn't match
    (defensive design for backward compatibility).

    Args:
        raw_shapes: Pre-transform shapes from dataset sample, one per entry.
        entries: DataEntry configs, each with optional transforms field.

    Returns:
        Post-transform shapes, one per entry.
    """
    if not entries or len(raw_shapes) != len(entries):
        return raw_shapes

    return tuple(
        _propagate_shape_through_chain(shape, getattr(entry, "transforms", ()) or ())
        for shape, entry in zip(raw_shapes, entries)
    )


def infer_post_transform_shapes(
    dataset: object,
    feature_entries: Sequence[Any],
    target_entries: Sequence[Any],
) -> ShapeSummary:
    """Infer post-transform shapes by propagating dataset shapes through transform chains.

    Combines raw shape inference from dataset with analytical shape propagation
    through configured transforms to compute shapes seen by the model.

    Args:
        dataset: Dataset object whose __getitem__ returns a nested TensorDict.
        feature_entries: Feature DataEntry configs with optional transforms field.
        target_entries: Target DataEntry configs with optional transforms field.

    Returns:
        ShapeSummary with post-transform in_shapes and out_shapes.

    Raises:
        ValueError: If raw shape inference fails or a transform lacks a registered
            shape inference function.
    """
    raw = infer_shapes_from_dataset(dataset)
    return ShapeSummary(
        in_shapes=_propagate_entry_shapes(raw.in_shapes, feature_entries),
        out_shapes=_propagate_entry_shapes(raw.out_shapes, target_entries),
    )
