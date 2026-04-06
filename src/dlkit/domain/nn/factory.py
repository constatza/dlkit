"""Pure factory function for model construction.

Replaces the _create_abc_model method in ProcessingLightningWrapper.
Uses explicit strategy selection via constructor signature inspection —
no ``contextlib.suppress`` fallback chains that mask user errors.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

    from dlkit.common.shapes import ShapeSummary


# Recognised aliases for input / output dimension injection.
_FFNN_IN_ALIASES: frozenset[str] = frozenset({"in_features", "input_dim", "input_size"})
_FFNN_OUT_ALIASES: frozenset[str] = frozenset({"out_features", "output_dim", "output_size"})


def _constructor_params(model_cls: type) -> frozenset[str]:
    """Return all non-self, non-variadic parameter names from ``__init__``.

    Args:
        model_cls: Model class to inspect.

    Returns:
        Frozenset of accepted parameter names (empty on inspection failure).
    """
    try:
        sig = inspect.signature(model_cls.__init__)
        return frozenset(
            name
            for name, p in sig.parameters.items()
            if name != "self"
            and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )
    except ValueError, TypeError:
        return frozenset()


def _is_lazy_module(model_cls: type) -> bool:
    """Return True if model_cls is a subclass of ``torch.nn.modules.lazy.LazyModuleMixin``.

    Args:
        model_cls: Model class to check.

    Returns:
        True if lazy, False otherwise (including when torch unavailable).
    """
    try:
        from torch.nn.modules.lazy import LazyModuleMixin

        return issubclass(model_cls, LazyModuleMixin)
    except ImportError:
        return False


def _lazy_kwargs(
    model_cls: type, shape: ShapeSummary, explicit_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Build kwargs for lazy models — inject only output-dim alias.

    Lazy modules infer input shapes on the first forward pass, so no
    input-dim injection is needed or correct.

    Args:
        model_cls: Lazy model class.
        shape: Shape summary from dataset.
        explicit_kwargs: User-provided hyperparameter kwargs.

    Returns:
        Merged kwargs dict with output-dim alias added if accepted.
    """
    params = _constructor_params(model_cls)
    kwargs: dict[str, Any] = dict(explicit_kwargs)
    out_alias = next((a for a in _FFNN_OUT_ALIASES if a in params), None)
    if out_alias and out_alias not in kwargs:
        kwargs[out_alias] = shape.out_features
    return kwargs


def _ffnn_kwargs(
    model_cls: type, shape: ShapeSummary, explicit_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Build kwargs for FFNN models — inject in_features / out_features aliases.

    Explicit user kwargs take precedence over shape-injected values.

    Args:
        model_cls: FFNN model class.
        shape: Shape summary from dataset.
        explicit_kwargs: User-provided hyperparameter kwargs.

    Returns:
        Merged kwargs dict with shape aliases added where accepted and absent.
    """
    params = _constructor_params(model_cls)
    kwargs: dict[str, Any] = dict(explicit_kwargs)
    in_alias = next((a for a in _FFNN_IN_ALIASES if a in params), None)
    out_alias = next((a for a in _FFNN_OUT_ALIASES if a in params), None)
    if in_alias and in_alias not in kwargs:
        kwargs[in_alias] = shape.in_features
    if out_alias and out_alias not in kwargs:
        kwargs[out_alias] = shape.out_features
    return kwargs


def _conv_kwargs(
    model_cls: type, shape: ShapeSummary, explicit_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Build kwargs for Conv/CAE models — inject in_channels / in_length.

    Explicit user kwargs take precedence over shape-injected values.

    Args:
        model_cls: Conv model class.
        shape: Shape summary from dataset.
        explicit_kwargs: User-provided hyperparameter kwargs.

    Returns:
        Merged kwargs dict with channel/length aliases added where accepted.
    """
    params = _constructor_params(model_cls)
    kwargs: dict[str, Any] = dict(explicit_kwargs)
    if "in_channels" not in kwargs:
        kwargs["in_channels"] = shape.in_channels
    if "in_length" in params and "in_length" not in kwargs:
        kwargs["in_length"] = shape.in_length
    return kwargs


def _select_strategy(model_cls: type) -> str:
    """Select construction strategy by inspecting the constructor signature.

    Strategy precedence:

    1. ``"lazy"`` — model is a ``LazyModuleMixin``; skip input-dim injection.
    2. ``"ffnn"`` — constructor has ``in_features`` / ``input_dim`` / ``input_size``.
    3. ``"conv"`` — constructor has ``in_channels``.
    4. ``"explicit"`` — no shape aliases found; use explicit kwargs only.

    Args:
        model_cls: Model class to inspect.

    Returns:
        Strategy name string.
    """
    if _is_lazy_module(model_cls):
        return "lazy"
    params = _constructor_params(model_cls)
    if params & _FFNN_IN_ALIASES:
        return "ffnn"
    if "in_channels" in params:
        return "conv"
    return "explicit"


def build_model(
    model_cls: type[nn.Module],
    shape: ShapeSummary | None,
    kwargs: dict[str, Any],
) -> nn.Module:
    """Construct a model using explicit strategy selection.

    Inspects the constructor signature once to select a single construction
    strategy. No ``contextlib.suppress`` — a misspelled kwarg raises
    ``TypeError`` immediately with the full error message.

    Strategies (in priority order):

    1. **No shape**: ``shape is None`` → pass ``kwargs`` only.
    2. **Lazy**: ``LazyModuleMixin`` → inject output-dim alias only.
    3. **FFNN**: constructor has ``in_features`` / ``input_dim`` / ``input_size``
       → inject input + output dim aliases.
    4. **Conv**: constructor has ``in_channels`` → inject channel/length aliases.
    5. **Explicit**: no recognised shape aliases → pass ``kwargs`` only.

    Explicit user ``kwargs`` always take precedence over shape-injected values.

    Args:
        model_cls: Model class to instantiate.
        shape: Shape summary from dataset inference, or ``None`` for
            external / shape-agnostic models.
        kwargs: Additional keyword arguments from model settings
            (already filtered to exclude structural fields).

    Returns:
        Constructed model instance.

    Raises:
        TypeError: If the model constructor rejects the provided arguments
            (e.g. unknown kwargs, wrong types). No silent fallback.
    """
    if shape is None:
        return model_cls(**kwargs)

    strategy = _select_strategy(model_cls)

    match strategy:
        case "lazy":
            final_kwargs = _lazy_kwargs(model_cls, shape, kwargs)
        case "ffnn":
            final_kwargs = _ffnn_kwargs(model_cls, shape, kwargs)
        case "conv":
            final_kwargs = _conv_kwargs(model_cls, shape, kwargs)
        case _:
            final_kwargs = dict(kwargs)

    return model_cls(**final_kwargs)
