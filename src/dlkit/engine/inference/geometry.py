"""Geometry inference for the inference subsystem.

Implements the 3-case fallback for restoring a GeometrySpec from a checkpoint
or a live dataset, replacing the old ShapeSummary-based shapes.py.
"""

from __future__ import annotations

from typing import Any

from dlkit.common.geometry import FieldRole, FieldSpec, GeometryKind, GeometrySpec, TopologyKind
from dlkit.infrastructure.utils.logging_config import get_logger

logger = get_logger(__name__)


def _geometry_from_dict(d: dict[str, Any]) -> GeometrySpec:
    """Reconstruct a GeometrySpec from a ``dataclasses.asdict()`` dict.

    Args:
        d: Dict produced by ``dataclasses.asdict(GeometrySpec)``.

    Returns:
        Restored GeometrySpec.
    """
    field_list = [
        FieldSpec(
            name=f["name"],
            shape=tuple(f["shape"]),
            role=FieldRole(f["role"]),
            geometry_kind=GeometryKind(f["geometry_kind"]),
        )
        for f in d.get("fields", [])
    ]
    topology_raw = d.get("topology_kind")
    return GeometrySpec(
        fields=tuple(field_list),
        topology_kind=TopologyKind(topology_raw) if topology_raw else None,
        edge_feature_dim=d.get("edge_feature_dim"),
    )


def _geometry_from_shape_summary(shape_data: dict[str, Any]) -> GeometrySpec | None:
    """Build a GeometrySpec from legacy ``shape_summary`` dict.

    Legacy format: ``{"in_shapes": [[d0, ...], ...], "out_shapes": [[d0, ...], ...]}``.
    Each element of ``in_shapes`` becomes one FEATURE FieldSpec.

    Args:
        shape_data: Dict with ``in_shapes`` and ``out_shapes`` lists.

    Returns:
        GeometrySpec if at least one in_shape is present, else None.
    """
    in_shapes = shape_data.get("in_shapes")
    if not in_shapes:
        return None
    field_specs = tuple(
        FieldSpec(
            name=f"in_{i}",
            shape=tuple(int(d) for d in s),
            role=FieldRole.FEATURE,
            geometry_kind=GeometryKind.TABULAR if len(s) == 1 else GeometryKind.REGULAR_GRID,
        )
        for i, s in enumerate(in_shapes)
    )
    return GeometrySpec(fields=field_specs)


def _infer_geometry_from_dataset(
    dataset: Any,
    entry_configs: tuple[Any, ...],
) -> GeometrySpec | None:
    """Derive a GeometrySpec by sampling dataset[0] and using entry configs.

    Args:
        dataset: Dataset whose ``__getitem__(0)`` returns a nested TensorDict
            with a ``"features"`` key.
        entry_configs: Feature DataEntry objects in config order.  If empty,
            falls back to raw shape extraction.

    Returns:
        GeometrySpec if at least one feature field can be resolved, else None.
    """
    if entry_configs:
        try:
            from dlkit.engine.data.geometry import infer_geometry

            feature_entries = tuple(
                e for e in entry_configs if getattr(e, "field_role", None) is not None
            )
            if feature_entries:
                return infer_geometry(feature_entries, dataset)
        except Exception as exc:
            logger.warning("Entry-config-guided geometry inference failed, falling back: {}", exc)

    # Bare fallback: extract raw shapes from dataset sample directly.
    try:
        from tensordict import TensorDictBase

        sample = dataset[0]
        if not isinstance(sample, TensorDictBase):
            return None
        feat_td = sample["features"]
        if not isinstance(feat_td, TensorDictBase):
            return None
        field_specs = tuple(
            FieldSpec(
                name=str(key),
                shape=tuple(int(d) for d in feat_td[key].shape),
                role=FieldRole.FEATURE,
                geometry_kind=GeometryKind.TABULAR,
            )
            for key in feat_td.keys()
        )
        if not field_specs:
            return None
        return GeometrySpec(fields=field_specs)
    except Exception:
        return None


def infer_geometry_from_checkpoint(
    checkpoint: dict[str, Any],
    dataset: Any | None = None,
    entry_configs: tuple[Any, ...] = (),
) -> GeometrySpec | None:
    """Infer a GeometrySpec using a 3-case fallback strategy.

    Case 1 — ``dlkit_metadata.geometry`` present: restore GeometrySpec from dict.
    Case 2 — ``dlkit_metadata`` absent (external checkpoint): return None.
    Case 3 — ``dlkit_metadata`` present but no geometry: try dataset fallback, else None.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        dataset: Optional dataset for shape inference fallback.
        entry_configs: Optional feature DataEntry objects for dataset fallback.

    Returns:
        GeometrySpec if geometry can be determined, otherwise None.
    """

    # Case 2: external checkpoint — no dlkit_metadata at all.
    if "dlkit_metadata" not in checkpoint:
        return None

    metadata = checkpoint["dlkit_metadata"]

    # Case 1: geometry already serialised in the checkpoint.
    geometry_data = metadata.get("geometry")
    if geometry_data:
        try:
            return _geometry_from_dict(geometry_data)
        except Exception as exc:
            logger.warning("Failed to restore GeometrySpec from checkpoint metadata: {}", exc)

    # TODO: remove shape_summary fallback once old checkpoints are no longer in circulation.
    # Legacy case: dlkit_metadata present with shape_summary (old format) → synthesise geometry.
    shape_data = metadata.get("shape_summary")
    if shape_data:
        try:
            geometry = _geometry_from_shape_summary(shape_data)
            if geometry is not None:
                return geometry
        except Exception as exc:
            logger.warning("Failed to build GeometrySpec from legacy shape_summary: {}", exc)

    # Case 3: dlkit_metadata present but no geometry → try dataset.
    if dataset is not None:
        try:
            result = _infer_geometry_from_dataset(dataset, entry_configs)
            if result is not None:
                return result
        except Exception as exc:
            logger.error("Dataset geometry inference failed: {}", exc)

    return None
