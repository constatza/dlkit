from __future__ import annotations

from typing import Literal

from torch import Tensor, nn

from dlkit.domain.nn.contracts import InputSpec as _InputSpec
from dlkit.domain.nn.contracts import StandardEntryConsumer
from dlkit.domain.nn.primitives.parametrized_layers import FactorizedLinear
from dlkit.domain.nn.utils import make_norm_layer


class LinearNetwork(StandardEntryConsumer, nn.Module):
    """A simple linear network with a single layer and optional normalization.

    This network consists of a single linear transformation with optional
    batch normalization or layer normalization.

    Args:
        in_features: Size of the input features.
        out_features: Size of the output features.
        normalize: Type of normalization to apply ('batch', 'layer', or None).
        bias: Whether to include bias in the linear layer.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        normalize: Literal["batch", "layer"] | None = None,
        bias: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.norm: nn.Module = make_norm_layer(normalize, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the linear network.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        x = self.linear(x)
        x = self.norm(x)
        return x


class FactorizedLinearNetwork(StandardEntryConsumer, nn.Module):
    """Single-layer network backed by one FactorizedLinear layer.

    The effective weight is ``W = diag(exp(log_scale)) @ base_weight`` with
    latent scales sampled directly from the paper-style RWF Gaussian.
    No normalization wrapper — contrast with LinearNetwork.

    Args:
        in_features: Size of the input features.
        out_features: Size of the output features.
        bias: Whether to include a bias term.
        mean: Mean of the Gaussian used to sample the latent scale parameter.
        std: Standard deviation for log-scale initialisation.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
    ) -> None:
        super().__init__()
        self.linear = FactorizedLinear(in_features, out_features, bias, mean=mean, std=std)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        return self.linear(x)
