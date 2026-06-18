"""Private shape-inference helpers for analytical transform propagation."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Sequence
from contextlib import suppress

from dlkit.infrastructure.config.transform_settings import TransformSettings

_DEFAULT_TRANSFORM_MODULE = "dlkit.domain.transforms"
_EXCLUDED_TRANSFORM_FIELDS = {"name", "module_path"}

type TransformKwargs = dict[
    str, int | float | str | bool | list[int] | list[float] | list[str] | None
]


def _resolve_transform_class(transform_settings: TransformSettings) -> type:
    """Resolve a transform class from its settings object.

    Args:
        transform_settings: Settings object carrying a ``name`` (str or type)
            and an optional ``module_path``.

    Returns:
        The transform class referenced by the settings.

    Raises:
        TypeError: If ``name`` is neither a ``str`` nor a ``type``.
    """
    name = getattr(transform_settings, "name", None)
    raw_path = getattr(transform_settings, "module_path", None)
    module_path: str = raw_path or _DEFAULT_TRANSFORM_MODULE
    if isinstance(name, type):
        return name
    if not isinstance(name, str):
        raise TypeError(f"Expected transform name to be str or type, got {type(name).__name__}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def _extract_transform_kwargs(transform_settings: TransformSettings) -> TransformKwargs:
    """Extract constructor kwargs from a transform settings object.

    Args:
        transform_settings: Settings object exposing ``model_dump()``.

    Returns:
        Mapping of keyword arguments, excluding ``name`` and ``module_path``.
        An empty dict is returned when ``model_dump()`` is unavailable.
    """
    with suppress(AttributeError):
        return {
            k: v
            for k, v in transform_settings.model_dump().items()
            if k not in _EXCLUDED_TRANSFORM_FIELDS
        }
    return {}


def _propagate_shape_through_chain(
    shape: tuple[int, ...],
    transform_settings_list: Sequence[TransformSettings],
    *,
    leading_axes: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Propagate a sample shape through an ordered transform chain.

    Args:
        shape: Raw per-sample shape (excluding any synthetic leading axes).
        transform_settings_list: Ordered transform settings to apply.
        leading_axes: Synthetic leading axes prepended before propagation and
            stripped by value afterwards (e.g. a placeholder batch axis).

    Returns:
        The propagated shape with synthetic leading axes removed.

    Raises:
        ValueError: If a transform lacks an ``infer_output_shape`` method.
    """
    current = (*leading_axes, *shape)
    for ts in transform_settings_list:
        transform_cls = _resolve_transform_class(ts)
        all_kwargs = _extract_transform_kwargs(ts)
        valid_params = set(inspect.signature(transform_cls.__init__).parameters) - {"self"}
        kwargs = {k: v for k, v in all_kwargs.items() if k in valid_params}
        instance = transform_cls(**kwargs)
        if not hasattr(instance, "infer_output_shape"):
            raise ValueError(
                f"Transform '{getattr(ts, 'name', transform_cls)}' has no "
                "infer_output_shape() method. Cannot propagate shape analytically."
            )
        current = instance.infer_output_shape(current)
    if not leading_axes:
        return current
    remaining = list(current)
    for axis_val in reversed(sorted(leading_axes)):
        with suppress(ValueError):
            axis_index = remaining.index(axis_val)
            remaining.pop(axis_index)
    return tuple(remaining)
