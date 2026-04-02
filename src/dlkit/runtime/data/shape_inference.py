"""Runtime shape inference from TensorDict dataset samples and transform chains."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from dlkit.shared.shapes import ShapeSummary

if TYPE_CHECKING:
    from collections.abc import Sequence


def infer_shapes_from_dataset(dataset: Any) -> ShapeSummary:
    """Sample index 0 from a dataset and extract raw feature/target shapes."""

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


def infer_post_transform_shapes(
    dataset: object,
    feature_entries: Sequence[Any],
    target_entries: Sequence[Any],
) -> ShapeSummary:
    """Infer post-transform shapes seen by the model."""

    raw = infer_shapes_from_dataset(dataset)
    return ShapeSummary(
        in_shapes=_propagate_entry_shapes(raw.in_shapes, feature_entries),
        out_shapes=_propagate_entry_shapes(raw.out_shapes, target_entries),
    )


def _resolve_transform_class(transform_settings: Any) -> type:
    name = getattr(transform_settings, "name", None)
    raw_path = getattr(transform_settings, "module_path", None)
    module_path: str = raw_path or "dlkit.domain.transforms"

    if isinstance(name, type):
        return name
    if not isinstance(name, str):
        raise TypeError(f"Expected transform name to be a str or type, got {type(name).__name__}")

    import importlib

    module = importlib.import_module(module_path)
    return cast(type, getattr(module, name))


def _extract_transform_kwargs(transform_settings: Any) -> dict[str, Any]:
    exclude = {"name", "module_path"}
    with suppress(AttributeError):
        return {k: v for k, v in transform_settings.model_dump().items() if k not in exclude}
    return {}


def _propagate_shape_through_chain(
    shape: tuple[int, ...],
    transform_settings_list: Sequence[Any],
) -> tuple[int, ...]:
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


def _propagate_entry_shapes(
    raw_shapes: tuple[tuple[int, ...], ...],
    entries: Sequence[Any],
) -> tuple[tuple[int, ...], ...]:
    if not entries or len(raw_shapes) != len(entries):
        return raw_shapes

    return tuple(
        _propagate_shape_through_chain(shape, getattr(entry, "transforms", ()) or ())
        for shape, entry in zip(raw_shapes, entries, strict=True)
    )
