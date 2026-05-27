"""Gated MLP with pluggable gating mechanisms.

Implements :class:`GatedMLP`, a feed-forward network where each hidden
layer is a gating unit drawn from a user-supplied factory.  The raw input
``x`` is forwarded as context into every gate, allowing gating mechanisms
such as :class:`~dlkit.domain.nn.primitives.gated.GRNGate` or
:class:`~dlkit.domain.nn.primitives.gated.UVGate` to modulate hidden
states against the original features.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self, TypeVar

from torch import Tensor, nn

from dlkit.domain.nn.contracts import ModelContractSpec, TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import _resolve_hidden_size
from dlkit.domain.nn.types import NormalizerName
from dlkit.domain.nn.utils import make_norm_layer

_GateT = TypeVar("_GateT", bound=nn.Module)


class GatedMLP(nn.Module):
    """Feed-forward network with per-layer pluggable gating.

    Each hidden layer applies a gating mechanism (e.g. GLU or SwiGLU)
    followed by optional normalisation and dropout.  The original input
    ``x`` is forwarded as a context argument to every gate, allowing
    context-sensitive gates (GRN, UV) to modulate the hidden state against
    the raw features.

    Architecture (forward pass)::

        h = embedding(x)  # linear projection, no activation
        for gate, norm, drop in zip(gates, norms, drops):
            h = drop(norm(gate(h, x)))
        return output(h)

    Args:
        in_features: Dimensionality of the input.
        out_features: Dimensionality of the output.
        hidden_size: Width of all hidden layers.
        num_layers: Number of gated hidden layers (>= 1).
        gate_factory: Zero-argument callable returning an
            :class:`~dlkit.domain.nn.primitives.gated.IGatingMechanism`
            that is also an :class:`~torch.nn.Module`.  Called once per
            layer.  The returned gate's ``forward(h, x)`` receives ``x``
            of shape ``(batch, in_features)``, so context-sensitive gates
            (e.g. :class:`~dlkit.domain.nn.primitives.gated.GRNGate` with
            ``context_size=None``, or
            :class:`~dlkit.domain.nn.primitives.gated.UVGate`) must be
            constructed with the matching ``in_features`` or
            ``context_size``.
        normalize: Optional normalisation to apply after each gate.
            One of ``"batch"``, ``"layer"``, ``"instance"``, ``"none"``,
            or ``None``.
        dropout: Dropout probability applied after normalisation.
            ``0.0`` disables dropout.

    Raises:
        ValueError: If ``num_layers < 1``.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        gate_factory: Callable[[], _GateT],
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__()
        self.embedding = nn.Linear(in_features, hidden_size)
        self.gates = nn.ModuleList([gate_factory() for _ in range(num_layers)])
        self.norms = nn.ModuleList(
            [make_norm_layer(normalize, hidden_size) for _ in range(num_layers)]
        )
        self.drops = nn.ModuleList(
            [nn.Dropout(dropout) if dropout > 0.0 else nn.Identity() for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the gated MLP.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        h = self.embedding(x)
        for gate, norm, drop in zip(self.gates, self.norms, self.drops, strict=True):
            h = drop(norm(gate(h, x)))
        return self.output(h)

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Build a :class:`GatedMLP` from a model contract spec.

        Args:
            contract: A ModelContractSpec variant; must be TabulaRSpec.
            **kwargs: Additional keyword arguments forwarded to
                :meth:`__init__`.

        Returns:
            Constructed :class:`GatedMLP`.
        """
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )
