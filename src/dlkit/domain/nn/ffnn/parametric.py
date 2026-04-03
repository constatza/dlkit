from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn

from dlkit.domain.nn.primitives import SkipConnection
from dlkit.domain.nn.utils import make_norm_layer


class ParametricDenseBlock(nn.Module):
    """Dense block using a caller-supplied layer factory for the linear transformation.

    Pre-activation order mirrors DenseBlock: norm → activation → layer → dropout.

    Args:
        size: Square feature size (input == output).
        layer_factory: Callable that takes an integer size and returns an nn.Module.
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        *,
        size: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = size
        self.out_features = size
        self.norm = make_norm_layer(normalize, size)
        self.activation = activation
        self.layer = layer_factory(size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: norm → activation → layer → dropout.

        Args:
            x: Input tensor of shape (batch_size, size).

        Returns:
            Output tensor of shape (batch_size, size).
        """
        x = self.norm(x)
        x = self.activation(x)
        x = self.layer(x)
        return self.dropout(x)


class ConstantWidthParametricFFNN(nn.Module):
    """Constant-width network using parametric dense blocks (in == out == size).

    Args:
        size: Square feature size for all layers.
        num_layers: Number of parametric blocks.
        layer_factory: Callable[[int], nn.Module] that builds one parametric linear layer.
        residual: Whether to wrap each block in a skip connection.
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be a positive integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")

        super().__init__()

        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block = ParametricDenseBlock(
                size=size,
                layer_factory=layer_factory,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            )
            blocks.append(SkipConnection(block, layer_type="linear") if residual else block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Sequential forward pass through all blocks.

        Args:
            x: Input tensor of shape (batch_size, size).

        Returns:
            Output tensor of shape (batch_size, size).
        """
        for block in self.blocks:
            x = block(x)
        return x


class EmbeddedParametricFFNN(nn.Module):
    """Parametric network with linear input embedding and output projection.

    Architecture: Linear(in→hidden) → ConstantWidthParametricFFNN → Linear(hidden→out).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Size of the parametric body (square).
        num_layers: Number of parametric blocks.
        layer_factory: Callable[[int], nn.Module] for parametric layer construction.
        residual: Whether to use skip connections in the body.
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Linear(in_features, hidden_size)
        self.body = ConstantWidthParametricFFNN(
            size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.regression_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: embedding → body → regression.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        x = self.embedding_layer(x)
        x = self.body(x)
        return self.regression_layer(x)
