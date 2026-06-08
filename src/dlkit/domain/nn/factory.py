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
    1. If ``contract`` is provided and ``model_cls`` implements ``from_contract``,
       call ``model_cls.from_contract(contract, **kwargs)``.
    2. Fall through to plain ``model_cls(**kwargs)``.

    Args:
        model_cls: A model class or callable that returns an ``nn.Module``.
        kwargs: Keyword arguments forwarded to the constructor or factory method.
        contract: Optional lean shape-bundle contract.

    Returns:
        A fully constructed ``nn.Module``.

    Raises:
        WorkflowError: If fallback construction fails and the model expects a contract.
    """
    resolved_kwargs: dict[str, Any] = kwargs or {}

    is_consumer = isinstance(model_cls, type) and hasattr(model_cls, "from_contract")

    if contract is not None and is_consumer:
        consumer_cls = cast(type[ContractConsumer], model_cls)
        return cast("nn.Module", consumer_cls.from_contract(contract, **resolved_kwargs))

    try:
        return model_cls(**resolved_kwargs)
    except TypeError as exc:
        if is_consumer and contract is None:
            from dlkit.common.errors import WorkflowError

            raise WorkflowError(
                f"Failed to build {model_cls.__name__} from kwargs. This model expects a "
                "contract (e.g. from dataset geometry inference), but no contract was provided. "
                "Check your DATASET configuration (e.g. feature roles) and ensure contract "
                "inference succeeded.",
                {"error": str(exc), "model_cls": model_cls.__name__},
            ) from exc
        raise


__all__ = [
    "ModelFactory",
    "build_model",
]
