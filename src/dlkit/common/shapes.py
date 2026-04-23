"""Shared shape contracts used across layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True, kw_only=True)
class ShapeSummary:
    """Minimal shape info extracted from dataset samples."""

    in_shapes: tuple[tuple[int, ...], ...]
    out_shapes: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        """Validate that shape tuples are non-empty."""
        if not self.in_shapes:
            raise ValueError(
                "ShapeSummary.in_shapes must be non-empty; "
                "in_features, in_channels, and in_length accessors depend on index 0"
            )
        if not self.out_shapes:
            raise ValueError(
                "ShapeSummary.out_shapes must be non-empty; "
                "out_features accessor depends on index 0"
            )

    @property
    def in_features(self) -> int:
        """Primary input feature size.

        Accesses index 0 of in_shapes; assumes in_shapes is non-empty.
        Validated in __post_init__.
        """
        return self.in_shapes[0][0]

    @property
    def out_features(self) -> int:
        """Primary output feature size."""
        return self.out_shapes[0][0]

    @property
    def in_channels(self) -> int:
        """Input channels for convolutional models."""
        return self.in_shapes[0][0]

    @property
    def in_length(self) -> int:
        """Input length for convolutional and sequence models."""
        return self.in_shapes[0][1]


class ShapeSpecProtocol(Protocol):
    """Minimal shape-spec contract shared by models and transforms."""

    def get_input_shape(self) -> tuple[int, ...] | None:
        """Get the primary input shape."""
        ...

    def get_output_shape(self) -> tuple[int, ...] | None:
        """Get the primary output shape."""
        ...

    def get_shape(self, name: str) -> tuple[int, ...] | None:
        """Get shape for a specific named entry."""
        ...

    def has_shape(self, name: str) -> bool:
        """Check whether a named entry exists."""
        ...

    def get_all_shapes(self) -> dict[str, tuple[int, ...]]:
        """Get every available shape."""
        ...

    def is_empty(self) -> bool:
        """Check whether the shape spec is empty."""
        ...

    def model_family(self) -> str:
        """Get the model-family identifier."""
        ...

    def get_shape_data(self) -> Any:
        """Return the underlying rich shape payload."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shape spec."""
        ...

    def with_canonical_aliases(self) -> ShapeSpecProtocol:
        """Return a spec including canonical aliases when available."""
        ...
