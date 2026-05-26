from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

from torch import Tensor, nn

from dlkit.domain.nn.contracts import ModelContractSpec, TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import _resolve_hidden_size
from dlkit.domain.nn.primitives import DenseBlock, SkipConnection, build_linear_skip_layer


class FeedForwardNN(nn.Module):
    """Feed-forward neural network with residual skip connections."""

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
            self.layers.append(SkipConnection(block, build_linear_skip_layer(block, bias=bias)))

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


class ConstantWidthFFNN(FeedForwardNN):
    """Residual feed-forward network with constant-width hidden layers.

    Note:
        ``num_layers`` is the number of hidden width entries passed to
        :class:`FeedForwardNN`.  The number of nonlinear ``DenseBlock`` layers
        created is ``num_layers - 1`` (transitions between adjacent entries).
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
        )
