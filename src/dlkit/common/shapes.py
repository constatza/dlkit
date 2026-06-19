"""Shape type aliases and contracts for the DLKit data layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

type Shape = tuple[int, ...]
"""Shape of a single tensor sample, e.g. ``(64, 128)``."""

type InputShapes = Mapping[str, Shape]
"""Read-only mapping from feature name to its shape."""

type OutputShapes = Mapping[str, Shape]
"""Read-only mapping from target name to its shape."""


@runtime_checkable
class ShapeProvider(Protocol):
    """Anything that can answer: what shape does entry X have?"""

    def get_shape(self, entry_name: str) -> Shape | None: ...


@dataclass(frozen=True)
class ShapeContext:
    """Immutable IO shape container implementing ShapeProvider.

    Args:
        input_shapes: Mapping from feature entry name to its shape.
        output_shapes: Mapping from target entry name to its shape.
    """

    input_shapes: InputShapes
    output_shapes: OutputShapes

    def get_shape(self, entry_name: str) -> Shape | None:
        """Return the shape for a named entry, searching inputs then outputs.

        Args:
            entry_name: Entry name to look up.

        Returns:
            Shape tuple if found, ``None`` otherwise.
        """
        if entry_name in self.input_shapes:
            return self.input_shapes[entry_name]
        return self.output_shapes.get(entry_name)


__all__ = ["InputShapes", "OutputShapes", "Shape", "ShapeContext", "ShapeProvider"]
