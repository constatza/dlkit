"""Pure function for building a GeometrySpec from feature entry configs and a dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from dlkit.common.geometry import FieldSpec, GeometryKind, GeometrySpec, TopologyKind
from dlkit.engine.data.shape_inference import (
    _propagate_shape_through_chain,
)

if TYPE_CHECKING:
    from dlkit.infrastructure.config.data_entries import DataEntry


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
