"""Pure reflection helpers for validating named forward()-kwarg contracts.

No framework dependency (no torch, no pydantic) — safe to import from both
``dlkit.domain`` and ``dlkit.engine`` per ``tach.toml`` (both may depend on
``dlkit.common``; neither may depend on the other).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ForwardSignatureCheck:
    """Facts about a callable's signature relevant to named-kwarg dispatch safety.

    Attributes:
        var_positional: Name of the ``*args`` parameter, if declared.
        var_keyword: Name of the ``**kwargs`` parameter, if declared.
        allowed_kwargs: Parameter names that accept keyword dispatch.
        unknown: Subset of the candidate kwarg names absent from the signature.
    """

    var_positional: str | None
    var_keyword: str | None
    allowed_kwargs: frozenset[str]
    unknown: frozenset[str]


def check_forward_kwargs(
    func: Callable[..., object], kwarg_names: frozenset[str] | set[str]
) -> ForwardSignatureCheck:
    """Inspect ``func``'s signature for named-kwarg dispatch safety.

    Args:
        func: Callable to inspect (typically ``model.forward``, bound or
            unbound — unbound access includes ``self`` in ``allowed_kwargs``,
            which is harmless since callers only check subset membership).
        kwarg_names: Candidate kwarg names the caller intends to dispatch.

    Returns:
        ``ForwardSignatureCheck`` with variadic-parameter names (if any) and
        the subset of ``kwarg_names`` absent from the signature.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    var_positional = next(
        (p.name for p in params.values() if p.kind is inspect.Parameter.VAR_POSITIONAL), None
    )
    var_keyword = next(
        (p.name for p in params.values() if p.kind is inspect.Parameter.VAR_KEYWORD), None
    )
    allowed = frozenset(
        p.name
        for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
    return ForwardSignatureCheck(
        var_positional=var_positional,
        var_keyword=var_keyword,
        allowed_kwargs=allowed,
        unknown=frozenset(kwarg_names) - allowed,
    )


__all__ = ["ForwardSignatureCheck", "check_forward_kwargs"]
