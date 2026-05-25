"""Explicit shape-aware and contract-based model construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, Self, cast, runtime_checkable

if TYPE_CHECKING:
    from torch import nn

    from dlkit.common.shapes import ShapeSummary

from dlkit.domain.nn.contracts import ContractConsumer, ModelContractSpec


@runtime_checkable
class ShapeConsumer(Protocol):
    """Protocol for models that explicitly opt into shape-based construction."""

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        """Construct the model from a dataset-derived shape summary."""
        ...


type ModelFactory = type[nn.Module] | Callable[..., nn.Module]


def build_model(
    model_cls: ModelFactory,
    shape: ShapeSummary | None = None,
    kwargs: dict[str, Any] | None = None,
    *,
    contract: ModelContractSpec | None = None,
) -> nn.Module:
    """Construct a model with explicit opt-in shape or contract handling.

    Dispatch priority:
    1. If ``contract`` is provided and ``model_cls`` is a ``ContractConsumer``,
       call ``model_cls.from_contract(contract, **kwargs)``.
    2. If ``shape`` is provided and ``model_cls`` has a callable ``from_shape``,
       call ``model_cls.from_shape(shape, **kwargs)``.
    3. Fall through to plain ``model_cls(**kwargs)``.

    Args:
        model_cls: A model class or callable that returns an ``nn.Module``.
        shape: Optional dataset-derived shape summary (legacy path).
        kwargs: Keyword arguments forwarded to the constructor or factory method.
        contract: Optional lean shape-bundle contract (new path).

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

    if shape is not None:
        from_shape = getattr(model_cls, "from_shape", None)
        if callable(from_shape):
            return from_shape(shape, **resolved_kwargs)

    return model_cls(**resolved_kwargs)


__all__ = [
    "ModelFactory",
    "ShapeConsumer",
    "build_model",
]
