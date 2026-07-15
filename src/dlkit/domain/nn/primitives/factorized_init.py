from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor

from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolved_activation_name

_DEFAULT_LOG_SCALE_MEAN = 0.0
_DEFAULT_LOG_SCALE_STD = 0.1
_DEFAULT_LEAKY_RELU_A = 0.01


@dataclass(frozen=True)
class FactorizedInit:
    """Activation-derived initialization policy for factorized linear layers."""

    log_scale_mean: float
    log_scale_std: float
    kaiming_a: float


def resolve_kaiming_a(
    activation: ActivationName | str | Callable[[Tensor], Tensor] | None,
    default: ActivationName = "relu",
) -> float:
    """Resolve the Kaiming ``a`` parameter for factorized base weights.

    Factorized layers keep the log-scale distribution fixed at unit scale and
    vary only the base-weight gain from the activation. Symmetric activations
    fall back to conservative ReLU-family Kaiming because the factorized
    primitive has one Kaiming-based base-weight path.
    """

    name = resolved_activation_name(activation, default)
    if callable(activation) and name not in {
        "relu",
        "gelu",
        "silu",
        "none",
        "identity",
        "leaky_relu",
        "tanh",
        "sigmoid",
    }:
        name = default

    match name:
        case "leaky_relu":
            return _DEFAULT_LEAKY_RELU_A
        case "relu" | "gelu" | "silu" | "none" | "identity" | "tanh" | "sigmoid":
            return 0.0
        case _:
            raise ValueError(f"Unsupported activation: {name!r}")


def resolve_factorized_init(
    activation: ActivationName | str | Callable[[Tensor], Tensor] | None,
    default: ActivationName = "relu",
) -> FactorizedInit:
    """Return fixed unit-scale log-scale init plus activation-derived base gain."""

    return FactorizedInit(
        log_scale_mean=_DEFAULT_LOG_SCALE_MEAN,
        log_scale_std=_DEFAULT_LOG_SCALE_STD,
        kaiming_a=resolve_kaiming_a(activation, default=default),
    )


__all__ = ["FactorizedInit", "resolve_factorized_init", "resolve_kaiming_a"]
