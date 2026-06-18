"""Shape type aliases for the DLKit data layer."""

from __future__ import annotations

from collections.abc import Mapping

type Shape = tuple[int, ...]
"""Shape of a single tensor sample, e.g. ``(64, 128)``."""

type InputShapes = Mapping[str, Shape]
"""Read-only mapping from feature name to its shape."""

type OutputShapes = Mapping[str, Shape]
"""Read-only mapping from target name to its shape."""

type EntryShapes = tuple[InputShapes, OutputShapes]
"""Pair of ``(InputShapes, OutputShapes)`` describing a dataset's IO contract."""

__all__ = ["EntryShapes", "InputShapes", "OutputShapes", "Shape"]
