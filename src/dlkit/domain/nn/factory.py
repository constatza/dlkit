"""Entry-shape-based and plain model construction helpers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from dlkit.domain.nn.contracts import HyperParam

if TYPE_CHECKING:
    from torch import nn

    from dlkit.common.sources import InputShapes, OutputShapes
    from dlkit.domain.nn.contracts import EntryConsumer

type ModelFactory = type[nn.Module] | Callable[..., nn.Module]

_log = logging.getLogger(__name__)


def _filter_to_accepted_kwargs(
    model_cls: ModelFactory, kwargs: dict[str, HyperParam]
) -> dict[str, HyperParam]:
    """Return only kwargs that model_cls's constructor will accept.

    ``**kwargs`` is only treated as "accept everything" when the class defines
    its **own** ``__init__``.  Inherited ``**kwargs`` (e.g. ``nn.Module``'s
    validation trap) are not trusted, so only explicitly named parameters pass.
    A warning is emitted for every dropped key so callers can diagnose mismatches.

    Args:
        model_cls: Model class or factory callable.
        kwargs: Candidate keyword arguments.

    Returns:
        Filtered dict of kwargs safe to pass to ``model_cls``.
    """
    try:
        sig = inspect.signature(model_cls)
        params = sig.parameters
        is_type = isinstance(model_cls, type)
        # Trust **kwargs when it's the model's own intent — a plain callable or a class
        # that defines its own __init__ with **kwargs.  An inherited **kwargs (e.g. the
        # validation trap in nn.Module.__init__) is not trusted.
        has_own_init = is_type and "__init__" in model_cls.__dict__
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        if has_var_keyword and (not is_type or has_own_init):
            return kwargs
        accepted = {k: v for k, v in kwargs.items() if k in params}
        dropped = set(kwargs) - set(accepted)
        if dropped:
            cls_name = getattr(model_cls, "__name__", repr(model_cls))
            _log.warning(
                f"build_model: {cls_name} does not accept {sorted(dropped)} — "
                "dropping from constructor call"
            )
        return accepted
    except ValueError, TypeError:
        return kwargs


def model_accepts_kwarg(model_cls: ModelFactory, kwarg: str) -> bool:
    """Return True if model_cls's constructor accepts the named keyword argument.

    Args:
        model_cls: Model class or factory callable.
        kwarg: Parameter name to check.

    Returns:
        True if the parameter exists in the constructor signature.
    """
    try:
        params = inspect.signature(model_cls).parameters
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        return kwarg in params or has_var_keyword
    except ValueError, TypeError:
        return False


def build_model(
    model_cls: ModelFactory,
    kwargs: dict[str, HyperParam] | None = None,
    *,
    input_shapes: InputShapes | None = None,
    output_shapes: OutputShapes | None = None,
) -> nn.Module:
    """Construct a model with explicit opt-in entry-shape handling.

    Dispatch priority:

    1. If both ``input_shapes`` and ``output_shapes`` are provided and
       ``model_cls`` implements ``from_entries``, call
       ``model_cls.from_entries(input_shapes, output_shapes, **kwargs)``.
    2. Fall through to plain ``model_cls(**kwargs)``.

    Args:
        model_cls: A model class or callable that returns an ``nn.Module``.
        kwargs: Keyword arguments forwarded to the constructor or factory method.
        input_shapes: Optional mapping from feature name to shape.
        output_shapes: Optional mapping from target name to shape.

    Returns:
        A fully constructed ``nn.Module``.

    Raises:
        WorkflowError: If fallback construction fails and the model expects
            entry shapes that were not provided.
    """
    raw_kwargs: dict[str, HyperParam] = kwargs or {}
    is_consumer = isinstance(model_cls, type) and hasattr(model_cls, "from_entries")

    if input_shapes is not None and output_shapes is not None and is_consumer:
        consumer = cast("EntryConsumer", model_cls)
        return cast("nn.Module", consumer.from_entries(input_shapes, output_shapes, **raw_kwargs))

    # Plain construction — drop kwargs that __init__ won't accept.
    init_kwargs = _filter_to_accepted_kwargs(model_cls, raw_kwargs)
    try:
        return model_cls(**init_kwargs)
    except TypeError as exc:
        if is_consumer:
            from dlkit.common.errors import WorkflowError

            raise WorkflowError(
                f"Failed to build {model_cls.__name__} — input_shapes and output_shapes "
                "are required for this model but were not provided. Check your DATASET "
                "configuration and ensure shape inference succeeded.",
                {"error": str(exc), "model_cls": model_cls.__name__},
            ) from exc
        raise


__all__ = [
    "ModelFactory",
    "build_model",
    "model_accepts_kwarg",
]
