"""Concrete, optional base class for dlkit-native models.

See ``dlkit.domain.nn.contracts`` for the structural ``EntryConsumer``
Protocol and the ``InputSpec``/``OutputSpec``/``StandardEntryConsumer``
pieces this module composes.
"""

from __future__ import annotations

from abc import ABC, ABCMeta, abstractmethod
from typing import Any

from torch import nn

from dlkit.common.errors import ForwardContractError
from dlkit.common.forward_contract import check_forward_kwargs
from dlkit.domain.nn.contracts import StandardEntryConsumer


def validate_model_contract(cls: type) -> None:
    """Check a model class's ``InputSpec`` (if any) agrees with ``forward()``.

    Used by ``_DLKitModuleMeta`` at class-definition time for nominal
    ``DLKitModule`` subclasses. ``infrastructure.registry.public``'s
    structural ``register_model`` check performs the equivalent check for
    non-nominal third-party models at registration time, independently
    (it cannot import this module — ``infrastructure`` and ``domain`` are
    sibling layers per ``tach.toml``) — both share the underlying
    ``dlkit.common.forward_contract.check_forward_kwargs`` primitive instead.
    An absent or empty ``InputSpec`` is not itself an error here for the
    empty case (see ``DLKitModule`` docstring); callers that require
    ``InputSpec`` to be declared at all enforce that themselves.

    Args:
        cls: Model class to validate. Must have a ``forward`` attribute.

    Raises:
        ForwardContractError: If ``InputSpec`` declares a field name with no
            matching ``forward()`` parameter.
    """
    input_spec = cls.__dict__.get("InputSpec") or getattr(cls, "InputSpec", None)
    if input_spec is None:
        return
    field_names = frozenset(getattr(input_spec, "model_fields", {}))
    if not field_names:
        return  # single-flat-input convention — nothing named to check
    # forward is accessed unbound here (no instance exists yet at
    # class-definition time), so its signature includes "self" — excluded
    # from the reported allowed set purely for message clarity.
    check = check_forward_kwargs(getattr(cls, "forward"), field_names)  # noqa: B009
    allowed = check.allowed_kwargs - {"self"}
    if check.unknown:
        raise ForwardContractError(
            f"{cls.__qualname__}.InputSpec declares {sorted(check.unknown)} with no "
            f"matching forward() parameter. Available: {sorted(allowed)}"
        )


class _DLKitModuleMeta(ABCMeta):
    """Runs the InputSpec/forward()-name contract check on every concrete subclass.

    Must be a metaclass hook, not ``__init_subclass__``: ``__abstractmethods__``
    is computed by ``ABCMeta.__new__`` *after* ``super().__new__()`` returns,
    but ``__init_subclass__`` fires *inside* ``type.__new__`` — before it's
    set. A metaclass ``__new__`` hook runs after ``super().__new__()``, where
    ``__abstractmethods__`` is already accurate, so still-abstract
    intermediate bases are correctly skipped.
    """

    def __new__(
        mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any
    ) -> type:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if getattr(cls, "__abstractmethods__", None):
            return cls  # still abstract — nothing to validate yet
        if getattr(cls, "InputSpec", None) is None:
            raise ForwardContractError(
                f"{cls.__qualname__} is a concrete DLKitModule but declares no "
                "InputSpec. Add `class InputSpec(InputSpec): pass` for a single "
                "unambiguous input, or declare named fields matching your "
                "forward() parameters.",
                {"model": cls.__qualname__},
            )
        validate_model_contract(cls)
        return cls


class DLKitModule(StandardEntryConsumer, nn.Module, ABC, metaclass=_DLKitModuleMeta):
    """Concrete, optional, recommended base for dlkit-native models.

    Behaves exactly like ``nn.Module`` in every way that matters — same
    ``__init__``, ``forward``, hooks, ``.to()``/``.state_dict()``,
    ``torch.compile`` support — plus a required, checked ``InputSpec``
    declaring which ``forward()`` parameters are named entry inputs.

    Not required to use dlkit's training/build pipeline (that only needs the
    structural ``EntryConsumer`` protocol via ``hasattr(cls, "from_context")``)
    or the registry (``register_model`` checks the same contract
    structurally, via plain attribute access, not nominal inheritance) — this
    is the opt-in path for the stronger, class-definition-time guarantee.
    """

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...


__all__ = ["DLKitModule", "validate_model_contract"]
