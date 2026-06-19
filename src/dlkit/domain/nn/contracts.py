"""Entry-consumer protocol and spec bases for DLKit models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from dlkit.common.shapes import InputShapes, OutputShapes, ShapeContext

type HyperParam = int | float | str | bool | list[int] | list[float] | list[str] | None


class InputSpec(BaseModel):
    """Base for per-model input entry-name declaration.

    Field names correspond to the entry names a model family expects. Extra
    fields are permitted so that subclasses may declare only the names they
    care about.
    """

    model_config = {"extra": "allow"}


class OutputSpec(BaseModel):
    """Base for per-model output entry-name declaration.

    Field names correspond to the target entry names this model produces.
    Mirrors InputSpec. Extra fields permitted so subclasses declare only
    the entries they own.
    """

    model_config = {"extra": "allow"}


@runtime_checkable
class EntryConsumer(Protocol):
    """Protocol for models constructible from named entry shapes."""

    InputSpec: type[InputSpec]

    @classmethod
    def shape_kwarg_names(cls) -> frozenset[str]:
        """Static declaration of which constructor kwargs are shape-derived.

        Callable without a ShapeContext — enables tooling and checkpoint
        separation without building the model.

        Returns:
            Frozenset of constructor kwarg names supplied by ``from_context``.
        """
        ...

    @classmethod
    def resolve_shape_kwargs(cls, context: ShapeContext) -> dict[str, int]:
        """Compute shape-derived constructor kwargs for a given context.

        Keys must match ``shape_kwarg_names()``.

        Args:
            context: Shape context carrying input and output shapes.

        Returns:
            Dict mapping constructor kwarg names to integer values.
        """
        ...

    @classmethod
    def from_context(
        cls,
        context: ShapeContext,
        **kwargs: HyperParam,
    ) -> Self:
        """Build the model from a ShapeContext plus hyper kwargs.

        Args:
            context: Shape context carrying input and output shapes.
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
    """Mixin: ``from_context`` is sealed; override ``resolve_shape_kwargs`` for non-flat IO.

    ``_SHAPE_KWARG_NAMES``: frozenset of constructor kwarg names that
    ``resolve_shape_kwargs`` supplies. Override when your ``resolve_shape_kwargs``
    returns non-standard keys.
    """

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_features", "out_features"})

    @classmethod
    def shape_kwarg_names(cls) -> frozenset[str]:
        """Return the set of constructor kwargs derived from shapes.

        Returns:
            Frozenset of kwarg names supplied by ``resolve_shape_kwargs``.
        """
        return cls._SHAPE_KWARG_NAMES

    @classmethod
    def resolve_shape_kwargs(cls, context: ShapeContext) -> dict[str, int]:
        """Map entry shapes to constructor kwargs via InputSpec/OutputSpec names.

        Uses the first field of ``InputSpec`` (if declared) as the feature entry
        name for a named lookup; falls back to positional order otherwise.

        Args:
            context: Shape context carrying input and output shapes.

        Returns:
            Dict with ``in_features`` and ``out_features`` keys.

        Raises:
            ValueError: If a required entry is absent from the context.
        """
        input_spec: type[InputSpec] | None = getattr(cls, "InputSpec", None)
        output_spec: type[OutputSpec] | None = getattr(cls, "OutputSpec", None)

        if input_spec and input_spec.model_fields:
            first_in = next(iter(input_spec.model_fields))
            in_shape = context.get_shape(first_in)
            if in_shape is None:
                raise ValueError(
                    f"{cls.__name__} requires input entry '{first_in}' "
                    f"but it is absent from ShapeContext (inputs: {list(context.input_shapes)})"
                )
        else:
            if len(context.input_shapes) != 1:
                raise ValueError(
                    f"{cls.__name__} has no InputSpec fields but ShapeContext has multiple "
                    f"input entries {list(context.input_shapes)}. "
                    "Declare InputSpec fields to specify which entry to use."
                )
            in_shape = next(iter(context.input_shapes.values()))

        if output_spec and output_spec.model_fields:
            first_out = next(iter(output_spec.model_fields))
            out_shape = context.get_shape(first_out)
            if out_shape is None:
                raise ValueError(
                    f"{cls.__name__} requires output entry '{first_out}' "
                    f"but it is absent from ShapeContext (outputs: {list(context.output_shapes)})"
                )
        else:
            if len(context.output_shapes) != 1:
                raise ValueError(
                    f"{cls.__name__} has no OutputSpec fields but ShapeContext has multiple "
                    f"output entries {list(context.output_shapes)}. "
                    "Declare OutputSpec fields to specify which entry to use."
                )
            out_shape = next(iter(context.output_shapes.values()))

        return {"in_features": in_shape[0], "out_features": out_shape[0]}

    @classmethod
    def from_context(
        cls,
        context: ShapeContext,
        **kwargs: HyperParam,
    ) -> Self:
        """Construct the model from a ShapeContext plus hyper kwargs.

        Args:
            context: Shape context carrying input and output shapes.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance of this model.

        Raises:
            ValueError: If required entries are missing from the context or if
                ``resolve_shape_kwargs`` key set diverges from ``shape_kwarg_names``.
        """
        input_spec: type[InputSpec] | None = getattr(cls, "InputSpec", None)
        if input_spec is not None and input_spec.model_fields:
            missing = set(input_spec.model_fields) - set(context.input_shapes)
            if missing:
                raise ValueError(
                    f"{cls.__name__} requires entries {sorted(missing)} "
                    f"but only {sorted(context.input_shapes)} are available"
                )
        shape_kwargs = cls.resolve_shape_kwargs(context)
        if frozenset(shape_kwargs) != cls.shape_kwarg_names():
            raise ValueError(
                f"{cls.__name__}.resolve_shape_kwargs() returned keys "
                f"{set(shape_kwargs)} but shape_kwarg_names() declares "
                f"{cls.shape_kwarg_names()} — keep them in sync"
            )
        return cls(**shape_kwargs, **kwargs)  # type: ignore[call-arg]


class SquareEntryConsumer(StandardEntryConsumer):
    """Mixin for models requiring in_features == out_features.

    Only ``in_features`` is derived from shapes; ``out_features`` is not
    supplied by this mixin (the constructor must not require it, or must
    default it).
    """

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_features"})

    @classmethod
    def resolve_shape_kwargs(cls, context: ShapeContext) -> dict[str, int]:
        """Validate square IO and return in_features.

        Args:
            context: Shape context carrying input and output shapes.

        Returns:
            Dict with ``in_features`` key only (in == out is validated).

        Raises:
            ValueError: If input and output shapes differ.
        """
        return {
            "in_features": _square_input_features(
                cls.__name__, context.input_shapes, context.output_shapes
            )
        }


__all__ = [
    "EntryConsumer",
    "HyperParam",
    "InputSpec",
    "OutputSpec",
    "SquareEntryConsumer",
    "StandardEntryConsumer",
    "_square_input_features",
]
