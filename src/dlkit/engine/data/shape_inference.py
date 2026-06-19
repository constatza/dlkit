"""Shape inference for entry-based datasets.

Replaces ``engine/data/geometry.py``. Returns plain shape dicts
(no ``GeometrySpec`` wrappers).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from tensordict import TensorDictBase

from dlkit.common.shapes import Shape, ShapeContext
from dlkit.infrastructure.precision.service import PrecisionService

from ._shape_helpers import _propagate_shape_through_chain

if TYPE_CHECKING:
    from dlkit.infrastructure.config.data_entries import DataEntry

_PLACEHOLDER_BATCH_AXIS = (1_000_003,)


def _require_tensordict_sample(sample: TensorDictBase, *, context: str) -> TensorDictBase:
    """Ensure a dataset sample is a nested ``TensorDict``.

    Args:
        sample: Candidate sample produced by a dataset.
        context: Human-readable caller name for error messages.

    Returns:
        The sample cast to ``TensorDictBase``.

    Raises:
        ValueError: If ``sample`` is not a ``TensorDictBase``.
    """
    if not isinstance(sample, TensorDictBase):
        raise ValueError(
            f"Expected {context} to produce a nested TensorDict sample, "
            f"got {type(sample).__name__}."
        )
    return sample


def _infer_shapes_for_entries(
    entries: Sequence[DataEntry],
    sub_td: TensorDictBase,
    *,
    leading_axes: tuple[int, ...],
    precision_service: PrecisionService,
) -> dict[str, Shape]:
    """Read raw shapes from ``sub_td`` and propagate per-entry transform chains.

    Tensors are precision-cast before shape extraction so that the reported
    shapes reflect the dtype the model will actually receive at runtime.

    Args:
        entries: Entries whose transform chains drive shape propagation.
        sub_td: Nested ``TensorDict`` keyed by entry name.
        leading_axes: Synthetic leading axes passed to transform propagation.
        precision_service: Shared ``PrecisionService`` used to cast each
            sampled tensor before its shape is read.

    Returns:
        Mapping from entry name to its post-transform shape.
    """
    result: dict[str, Shape] = {}
    entry_by_name = {e.name: e for e in entries if e.name is not None}
    for key in sub_td.keys():
        sample = precision_service.cast_tensor(cast(torch.Tensor, sub_td[key]))
        raw = tuple(int(d) for d in sample.shape)
        entry = entry_by_name.get(str(key))
        transforms = (getattr(entry, "transforms", ()) or ()) if entry else ()
        result[str(key)] = _propagate_shape_through_chain(
            raw, transforms, leading_axes=leading_axes
        )
    return result


def infer_entry_shapes(
    feature_entries: Sequence[DataEntry],
    target_entries: Sequence[DataEntry],
    sample: TensorDictBase,
) -> ShapeContext:
    """Infer post-transform shapes for all feature and target entries.

    Args:
        feature_entries: Feature ``DataEntry`` objects (model inputs).
        target_entries: Target ``DataEntry`` objects (loss inputs).
        sample: A single-sample ``TensorDict`` from the dataset (``batch_size=[]``).

    Returns:
        ``ShapeContext`` with ``input_shapes`` and ``output_shapes`` mappings.

    Raises:
        ValueError: If ``sample`` is not a ``TensorDictBase``.
    """
    sample_td = _require_tensordict_sample(sample, context="infer_entry_shapes()")
    precision_service = PrecisionService()

    input_shapes = _infer_shapes_for_entries(
        feature_entries,
        cast("TensorDictBase", sample_td["features"]),
        leading_axes=(),
        precision_service=precision_service,
    )
    target_shapes: dict[str, Shape] = {}
    if "targets" in sample_td.keys():
        target_shapes = _infer_shapes_for_entries(
            target_entries,
            cast("TensorDictBase", sample_td["targets"]),
            leading_axes=_PLACEHOLDER_BATCH_AXIS,  # ponytail: sentinel batch-axis for target shape propagation
            precision_service=precision_service,
        )
    return ShapeContext(input_shapes=input_shapes, output_shapes=target_shapes)


__all__ = ["infer_entry_shapes"]
