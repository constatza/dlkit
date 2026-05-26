"""Pure function for building a GeometrySpec from feature entry configs and a dataset."""

from __future__ import annotations

import importlib
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from dlkit.common.geometry import FieldSpec, GeometryKind, GeometrySpec, TopologyKind

if TYPE_CHECKING:
    from collections.abc import Sequence

    from dlkit.infrastructure.config.data_entries import DataEntry


def _resolve_transform_class(transform_settings: Any) -> type:
    """Resolve a transform class from transform settings.

    Args:
        transform_settings: Settings object with ``name`` and ``module_path`` attrs.

    Returns:
        The resolved transform class.

    Raises:
        TypeError: If ``name`` is neither a str nor a type.
    """
    name = getattr(transform_settings, "name", None)
    raw_path = getattr(transform_settings, "module_path", None)
    module_path: str = raw_path or "dlkit.domain.transforms"

    if isinstance(name, type):
        return name
    if not isinstance(name, str):
        raise TypeError(f"Expected transform name to be a str or type, got {type(name).__name__}")

    module = importlib.import_module(module_path)
    return cast(type, getattr(module, name))


def _extract_transform_kwargs(transform_settings: Any) -> dict[str, Any]:
    """Extract non-structural keyword arguments from transform settings.

    Args:
        transform_settings: Settings object with a ``model_dump()`` method.

    Returns:
        Dict of kwargs excluding ``name`` and ``module_path``.
    """
    exclude = {"name", "module_path"}
    with suppress(AttributeError):
        return {k: v for k, v in transform_settings.model_dump().items() if k not in exclude}
    return {}


def _propagate_shape_through_chain(
    shape: tuple[int, ...],
    transform_settings_list: Sequence[Any],
) -> tuple[int, ...]:
    """Propagate a shape through an ordered list of transform settings analytically.

    Each transform class must be registered in ``SHAPE_INFERENCE_REGISTRY``.

    Args:
        shape: Input shape to propagate.
        transform_settings_list: Ordered transform settings (each has ``name``
            and ``module_path`` attributes).

    Returns:
        Output shape after all transforms.

    Raises:
        ValueError: If a transform has no registered shape inference function.
    """
    from dlkit.domain.transforms.shape_inference import SHAPE_INFERENCE_REGISTRY

    current = shape
    for transform_settings in transform_settings_list:
        transform_cls = _resolve_transform_class(transform_settings)
        if transform_cls not in SHAPE_INFERENCE_REGISTRY:
            raise ValueError(
                f"Transform '{getattr(transform_settings, 'name', transform_cls)}' has no "
                "registered shape inference function. Specify explicit model init_kwargs "
                "to bypass analytical shape inference."
            )
        current = SHAPE_INFERENCE_REGISTRY[transform_cls](
            current,
            **_extract_transform_kwargs(transform_settings),
        )
    return current


def infer_geometry(
    feature_entries: tuple[DataEntry, ...],
    dataset: Any,
) -> GeometrySpec:
    """Build a GeometrySpec from feature entry configs and a dataset sample.

    Samples dataset[0] to get raw shapes, propagates each entry's transform
    chain analytically, then builds one FieldSpec per entry using the entry's
    field_role and geometry_kind.

    Args:
        feature_entries: Dataset-backed feature entry configs, in config order.
        dataset: Dataset with a __getitem__ that returns a TensorDict with a
                 "features" nested key at index 0.

    Returns:
        GeometrySpec with fields in feature_entries order.

    Raises:
        ValueError: If dataset[0] is not a nested TensorDict, or if a transform
                    has no registered shape inference function.
    """
    try:
        from tensordict import TensorDictBase
    except ImportError as exc:
        raise ImportError("tensordict is required for infer_geometry") from exc

    sample = dataset[0]

    if not isinstance(sample, TensorDictBase):
        raise ValueError(
            f"Expected dataset[0] to return a nested TensorDict with 'features', "
            f"got {type(sample).__name__}. Update your dataset's __getitem__ accordingly."
        )

    feat_td = cast(TensorDictBase, sample["features"])
    feature_keys = list(feat_td.keys())

    if len(feature_keys) != len(feature_entries):
        raise ValueError(
            f"feature_entries has {len(feature_entries)} entries but dataset[0]['features'] "
            f"has {len(feature_keys)} keys. They must match positionally."
        )

    field_specs = tuple(
        _build_field_spec(entry, feat_td[key])
        for entry, key in zip(feature_entries, feature_keys, strict=True)
    )

    topology_kind = (
        TopologyKind.EDGE_INDEX
        if any(spec.geometry_kind == GeometryKind.GRAPH for spec in field_specs)
        else None
    )

    return GeometrySpec(
        fields=field_specs,
        topology_kind=topology_kind,
        edge_feature_dim=None,
    )


def _build_field_spec(entry: DataEntry, tensor: Any) -> FieldSpec:
    """Build a single FieldSpec from a DataEntry and its raw tensor sample.

    Args:
        entry: Feature entry config carrying role, geometry_kind, and transforms.
        tensor: The raw tensor sample (shape includes sample dimensions, not batch).

    Returns:
        FieldSpec with shape after transform propagation.
    """
    if entry.name is None:
        raise ValueError(
            f"DataEntry of type {type(entry).__name__} has no name. "
            "All entries passed to infer_geometry() must have a name set."
        )
    raw_shape = tuple(int(d) for d in tensor.shape)
    transform_settings = getattr(entry, "transforms", ()) or ()
    post_shape = _propagate_shape_through_chain(raw_shape, transform_settings)
    return FieldSpec(
        name=entry.name,
        shape=post_shape,
        role=entry.field_role,
        geometry_kind=entry.geometry_kind,
    )
