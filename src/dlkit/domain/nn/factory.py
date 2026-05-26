"""Contract-based and plain model construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from torch import nn

from dlkit.domain.nn.contracts import ContractConsumer, ModelContractSpec

type ModelFactory = type[nn.Module] | Callable[..., nn.Module]


def build_model(
    model_cls: ModelFactory,
    kwargs: dict[str, Any] | None = None,
    *,
    contract: ModelContractSpec | None = None,
) -> nn.Module:
    """Construct a model with explicit opt-in contract handling.

    Dispatch priority:
    1. If ``contract`` is provided and ``model_cls`` is a ``ContractConsumer``,
       call ``model_cls.from_contract(contract, **kwargs)``.
    2. Fall through to plain ``model_cls(**kwargs)``.

    Args:
        model_cls: A model class or callable that returns an ``nn.Module``.
        kwargs: Keyword arguments forwarded to the constructor or factory method.
        contract: Optional lean shape-bundle contract.

    Returns:
        A fully constructed ``nn.Module``.
    """
    resolved_kwargs: dict[str, Any] = kwargs or {}

    if (
        contract is not None
        and isinstance(model_cls, type)
        and issubclass(model_cls, ContractConsumer)
    ):
        return cast("nn.Module", model_cls.from_contract(contract, **resolved_kwargs))

    return model_cls(**resolved_kwargs)


__all__ = [
    "ModelFactory",
    "build_model",
]
