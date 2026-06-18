"""Entry-consumer protocol and InputSpec base for DLKit models."""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel

from dlkit.common.sources import InputShapes, OutputShapes

type HyperParam = int | float | str | bool | list[int] | list[float] | list[str] | None


class InputSpec(BaseModel):
    """Base for per-model entry-name validation.

    Field names correspond to the entry names a model family expects. Extra
    fields are permitted so that subclasses may declare only the names they
    care about.
    """

    model_config = {"extra": "allow"}


@runtime_checkable
class EntryConsumer(Protocol):
    """Protocol for models constructible from named entry shapes."""

    InputSpec: type[InputSpec]

    @classmethod
    def from_entries(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
        **kwargs: HyperParam,
    ) -> Self:
        """Construct the model from named input and output shapes.

        Args:
            input_shapes: Mapping from feature entry name to its shape.
            output_shapes: Mapping from target entry name to its shape.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance of this model.
        """
        ...


__all__ = [
    "EntryConsumer",
    "HyperParam",
    "InputSpec",
]
