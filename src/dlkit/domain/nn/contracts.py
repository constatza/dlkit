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


def _square_input_features(
    cls_name: str,
    input_shapes: InputShapes,
    output_shapes: OutputShapes,
) -> int:
    """Validate square IO shapes and return the shared feature dimension.

    Args:
        cls_name: Name of the requesting class, used in error messages.
        input_shapes: Mapping from feature entry name to its shape.
        output_shapes: Mapping from target entry name to its shape.

    Returns:
        The leading dimension of the input shape (equal to the output's).

    Raises:
        ValueError: If the input and output shapes are not equal.
    """
    in_shape = next(iter(input_shapes.values()))
    out_shape = next(iter(output_shapes.values()))
    if in_shape != out_shape:
        raise ValueError(
            f"{cls_name} requires a square contract (in_shape == out_shape), "
            f"got in_shape={in_shape}, out_shape={out_shape}"
        )
    return in_shape[0]


class StandardEntryConsumer:
    """Mixin: from_entries is sealed; override _constructor_dims for non-flat IO.

    _SHAPE_KWARG_NAMES: frozenset of constructor kwarg names that _constructor_dims supplies.
    model_builder uses this to strip them from checkpoint hyperparams.
    Override when your _constructor_dims returns non-standard keys.
    """

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_features", "out_features"})

    @classmethod
    def _constructor_dims(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
    ) -> dict[str, int]:
        """Map entry shapes to constructor kwargs.

        Args:
            input_shapes: Mapping from feature entry name to its shape.
            output_shapes: Mapping from target entry name to its shape.

        Returns:
            Dict of constructor kwargs derived from shapes. Default: flat IO.
        """
        return {
            "in_features": next(iter(input_shapes.values()))[0],
            "out_features": next(iter(output_shapes.values()))[0],
        }

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

        Raises:
            ValueError: If required entries are missing from input_shapes.
        """
        input_spec: type[InputSpec] | None = getattr(cls, "InputSpec", None)
        if input_spec is not None and input_spec.model_fields:
            missing = set(input_spec.model_fields) - set(input_shapes)
            if missing:
                raise ValueError(
                    f"{cls.__name__} requires entries {sorted(missing)} "
                    f"but only {sorted(input_shapes)} are available"
                )
        return cls(**cls._constructor_dims(input_shapes, output_shapes), **kwargs)


class SquareEntryConsumer(StandardEntryConsumer):
    """Mixin for models requiring in_features == out_features.

    Only ``in_features`` is derived from shapes; ``out_features`` is not
    supplied by this mixin (the constructor must not require it, or must
    default it).
    """

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_features"})

    @classmethod
    def _constructor_dims(
        cls,
        input_shapes: InputShapes,
        output_shapes: OutputShapes,
    ) -> dict[str, int]:
        """Validate square IO and return in_features.

        Args:
            input_shapes: Mapping from feature entry name to its shape.
            output_shapes: Mapping from target entry name to its shape.

        Returns:
            Dict with ``in_features`` key only (in == out is validated).

        Raises:
            ValueError: If input and output shapes differ.
        """
        return {"in_features": _square_input_features(cls.__name__, input_shapes, output_shapes)}


__all__ = [
    "EntryConsumer",
    "HyperParam",
    "InputSpec",
    "SquareEntryConsumer",
    "StandardEntryConsumer",
    "_square_input_features",
]
