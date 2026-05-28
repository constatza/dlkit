from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

from torch import Tensor, nn

from dlkit.domain.nn.contracts import ModelContractSpec, TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import _resolve_hidden_size
from dlkit.domain.nn.primitives import DenseBlock, SkipConnection, build_linear_skip_layer


class VarWidthFFNN(nn.Module):
    """Feed-forward network with explicit per-layer widths.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        layers: Hidden-layer widths as an explicit list; each entry is one hidden layer.
        activation: Element-wise activation between layers.
        normalize: Optional normalisation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability between layers.
        bias: Whether linear layers include a bias term.
        skip: If ``True`` (default) each hidden transition is wrapped in a
            ``SkipConnection``; set to ``False`` for a plain dense network.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = len(layers)
        self.activation = activation

        self.layers = nn.ModuleList()
        self.embedding_layer = nn.Linear(in_features, layers[0], bias=bias)

        for i in range(self.num_layers - 1):
            block = DenseBlock(
                layers[i],
                layers[i + 1],
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
            )
            layer = (
                SkipConnection(block, build_linear_skip_layer(block, bias=bias)) if skip else block
            )
            self.layers.append(layer)

        self.regression_layer = nn.Linear(layers[-1], out_features, bias=bias)

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Build the network from a model contract spec.

        Args:
            contract: A ModelContractSpec variant; must be TabulaRSpec.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance.
        """
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression_layer(x)


class FFNN(VarWidthFFNN):
    """Feed-forward network with constant-width hidden layers.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        hidden_size: Width of all hidden layers; defaults to
            ``max(in_features, out_features)`` when omitted.
        num_layers: Number of hidden layers.
        activation: Element-wise activation between layers.
        normalize: Optional normalisation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability between layers.
        bias: Whether linear layers include a bias term.
        skip: If ``True`` (default) each hidden transition is wrapped in a
            ``SkipConnection``; set to ``False`` for a plain dense network.

    Note:
        The number of nonlinear ``DenseBlock`` layers created is
        ``num_layers - 1`` (transitions between adjacent entries).
        With ``num_layers=1`` the network is an embedding linear followed
        immediately by the regression linear — no nonlinearity.  Use
        ``num_layers >= 2`` for a genuinely nonlinear network.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        skip: bool = True,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=[hidden_size] * num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
            skip=skip,
        )
