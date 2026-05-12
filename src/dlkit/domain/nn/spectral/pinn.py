"""PINN-oriented frequency and coordinate encoding networks.

Implements three primitives from the physics-informed ML literature:

``FourierFeatureNetwork`` (Tancik et al. 2020)
    Projects each coordinate through a random or learned frequency matrix
    before the MLP, directly countering spectral bias.

``SirenFFNN`` (Sitzmann et al. 2020)
    MLP using ``sin`` activations with the initialisation from the original
    paper; particularly effective for PDE solutions and implicit representations.

``ModifiedMLP`` (Wang et al. 2022)
    Adds U/V encoder branches that gate hidden states; shown to accelerate
    PINN convergence significantly vs. standard MLP.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Self

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.ffnn.residual import ConstantWidthFFNN

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


class FourierFeatureNetwork(nn.Module):
    """MLP with coordinate-wise Fourier feature mapping (Tancik et al. 2020).

    Maps input coordinates through:

        γ(x) = [sin(2π B x), cos(2π B x)]  ∈ ℝ^{2m}

    where B ∈ ℝ^{m×d} is either frozen random or a learnable parameter.
    The encoded features are then passed through a residual MLP.

    Args:
        in_features: Dimension of the coordinate input (e.g. 2 for (x, t)).
        out_features: Dimension of the network output.
        hidden_size: Width of the hidden MLP layers.
        num_layers: Number of hidden MLP layers.
        n_frequencies: Number of frequency vectors m (encoding output is 2*m).
        sigma: Standard deviation for sampling B. Default 1.0.
        learnable_B: If True, B is an nn.Parameter; otherwise a frozen buffer.
        activation: Activation for the internal MLP.
        normalize: Normalisation type for the internal MLP.
        dropout: Dropout probability for the internal MLP.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        n_frequencies: int,
        sigma: float = 1.0,
        learnable_B: bool = False,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        B = sigma * torch.randn(n_frequencies, in_features)
        if learnable_B:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.mlp = ConstantWidthFFNN(
            in_features=2 * n_frequencies,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode coordinates and pass through MLP.

        Args:
            x: Coordinate tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        proj = 2 * math.pi * x @ self.B.T
        encoded = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return self.mlp(encoded)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary with in_features and out_features.
            **kwargs: Additional constructor arguments.

        Returns:
            Constructed FourierFeatureNetwork.
        """
        return cls(in_features=shape.in_features, out_features=shape.out_features, **kwargs)


class SirenFFNN(nn.Module):
    """Sinusoidal representation network (Sitzmann et al. 2020, NeurIPS).

    Uses sin activations throughout with layer-specific weight initialisation:

    * First layer: W ~ U(-1/d, 1/d) where d is the fan-in.
    * Subsequent layers: W ~ U(-sqrt(6/d), sqrt(6/d)).

    Args:
        in_features: Dimension of the input.
        out_features: Dimension of the output.
        hidden_size: Width of all hidden layers.
        num_layers: Number of hidden layers (each is sin + linear).
        omega0: Frequency multiplier for the first layer (default 30).
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        omega0: float = 30.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        super().__init__()
        self._omega0 = omega0

        self.first_layer = nn.Linear(in_features, hidden_size)
        nn.init.uniform_(self.first_layer.weight, -1.0 / in_features, 1.0 / in_features)
        nn.init.zeros_(self.first_layer.bias)

        bound = math.sqrt(6.0 / hidden_size) / omega0
        self.hidden_layers = nn.ModuleList(
            [self._make_hidden(hidden_size, bound) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, out_features)

    @staticmethod
    def _make_hidden(hidden_size: int, bound: float) -> nn.Linear:
        """Create a hidden linear layer with SIREN initialisation.

        Args:
            hidden_size: Width of the layer.
            bound: Uniform initialisation bound for weights.

        Returns:
            Initialised linear layer.
        """
        layer = nn.Linear(hidden_size, hidden_size)
        nn.init.uniform_(layer.weight, -bound, bound)
        nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with sinusoidal activations.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        x = torch.sin(self._omega0 * self.first_layer(x))
        for layer in self.hidden_layers:
            x = torch.sin(layer(x))
        return self.output_layer(x)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary with in_features and out_features.
            **kwargs: Additional constructor arguments.

        Returns:
            Constructed SirenFFNN.
        """
        return cls(in_features=shape.in_features, out_features=shape.out_features, **kwargs)


class ModifiedMLP(nn.Module):
    """Modified MLP with U/V encoder gating (Wang et al. 2022).

    Two learned encoder branches U and V modulate each hidden layer:

        U = σ(W_u x + b_u)
        V = σ(W_v x + b_v)
        h₀ = σ(W₀ x + b₀)
        zₖ = σ(Wₖ hₖ + bₖ)
        hₖ₊₁ = zₖ ⊙ U + (1 − zₖ) ⊙ V
        output = W_out h_L + b_out

    Reference: Wang et al., "Improved architectures and training algorithms
    for deep operator networks", J. Comp. Phys. 2022.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        hidden_size: Width of all hidden layers.
        num_layers: Number of hidden linear layers (>= 1).
        activation: Gating activation σ. Defaults to torch.sigmoid.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        activation: Callable[[Tensor], Tensor] = torch.sigmoid,
    ) -> None:
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 (U/V gating requires at least one hidden layer)"
            )
        super().__init__()
        self.activation = activation
        self.encoder_u = nn.Linear(in_features, hidden_size)
        self.encoder_v = nn.Linear(in_features, hidden_size)
        self.input_layer = nn.Linear(in_features, hidden_size)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with U/V gating.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        u = self.activation(self.encoder_u(x))
        v = self.activation(self.encoder_v(x))
        h = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            z = self.activation(layer(h))
            h = z * u + (1.0 - z) * v
        return self.output_layer(h)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary with in_features and out_features.
            **kwargs: Additional constructor arguments.

        Returns:
            Constructed ModifiedMLP.
        """
        return cls(in_features=shape.in_features, out_features=shape.out_features, **kwargs)
