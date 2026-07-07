"""Weight initialization resolved from the same activation name used at build time.

Mirrors ``resolve_activation``'s dispatch style so a module's stored
activation name also picks its weight init, without threading a second
scheme kwarg through every constructor.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor, nn

from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolved_activation_name

_KNOWN_ACTIVATION_NAMES = frozenset(
    {"relu", "gelu", "silu", "none", "identity", "leaky_relu", "tanh", "sigmoid"}
)

_LINEAR_AND_CONV = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def resolve_initializer(
    activation: ActivationName | str | Callable | None,
    default: ActivationName = "relu",
) -> Callable[[Tensor], Tensor]:
    """Resolve the in-place weight initializer matching an activation.

    Args:
        activation: Activation name, callable, or None (mirrors
            ``resolve_activation``'s input).
        default: Activation name used when ``activation`` is None.

    Returns:
        A callable that initializes a weight tensor in-place and returns it.

    Raises:
        ValueError: If the resolved activation name is not recognised.
    """
    name = resolved_activation_name(activation, default)
    if callable(activation) and name not in _KNOWN_ACTIVATION_NAMES:
        # Custom callables (e.g. lambdas) carry no recoverable name to key
        # off; fall back to the default scheme rather than failing
        # construction. Unrecognised *string* input still raises below.
        name = default
    match name:
        # gelu/silu have no dedicated gain in torch.nn.init.calculate_gain;
        # ReLU-family Kaiming is the accepted practical default for them.
        case "relu" | "gelu" | "silu" | "none" | "identity":
            return lambda w: nn.init.kaiming_uniform_(w, nonlinearity="relu")
        case "leaky_relu":
            return lambda w: nn.init.kaiming_uniform_(w, a=0.01, nonlinearity="leaky_relu")
        case "tanh":
            return lambda w: nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("tanh"))
        case "sigmoid":
            return lambda w: nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain("sigmoid"))
        case _:
            raise ValueError(f"Unsupported activation: {name!r}")


def initialize_(
    module: nn.Module,
    activation: ActivationName | str | Callable | None,
    default: ActivationName = "relu",
) -> None:
    """Apply the activation-matched initializer to Linear/Conv weights.

    Walks ``module.modules()`` (including ``module`` itself) and, for every
    ``nn.Linear``/``nn.Conv*``/``nn.ConvTranspose*`` submodule, initializes
    its weight with :func:`resolve_initializer` and zero-inits its bias.

    Args:
        module: Module (or single layer) to initialize in-place.
        activation: Activation name, callable, or None used to pick the scheme.
        default: Activation name used when ``activation`` is None.
    """
    init_weight = resolve_initializer(activation, default)
    for m in module.modules():
        if isinstance(m, _LINEAR_AND_CONV):
            init_weight(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
