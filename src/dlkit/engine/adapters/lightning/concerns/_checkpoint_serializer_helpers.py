"""Private serialization helpers for checkpoint metadata."""

from __future__ import annotations

from collections.abc import Mapping


def serialize_shapes(shapes: Mapping[str, tuple[int, ...]] | None) -> dict[str, list[int]] | None:
    """Convert an entry-shape mapping into a JSON-safe dict.

    Args:
        shapes: Mapping from entry name to a shape tuple, or ``None``.

    Returns:
        Dict mapping entry name to a list of ints, or ``None`` when ``shapes``
        is falsy.
    """
    if not shapes:
        return None
    return {str(name): [int(dim) for dim in shape] for name, shape in shapes.items()}


def deserialize_shapes(data: Mapping[str, list[int]] | None) -> dict[str, tuple[int, ...]] | None:
    """Reconstruct an entry-shape mapping from serialized checkpoint data.

    Args:
        data: Dict mapping entry name to a list of ints, or ``None``.

    Returns:
        Dict mapping entry name to a shape tuple, or ``None`` when ``data`` is
        falsy.
    """
    if not data:
        return None
    return {str(name): tuple(int(dim) for dim in shape) for name, shape in data.items()}
