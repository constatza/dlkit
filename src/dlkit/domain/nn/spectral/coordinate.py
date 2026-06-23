"""Coordinate spectral-bias networks.

Implements coordinate encoders and architectures commonly used to counter
spectral bias:

``FourierFeatureNetwork`` (Tancik et al. 2020)
    Projects each coordinate through a random or learned frequency matrix
    before the MLP, directly countering spectral bias.

``HashEncodingNetwork`` (in the style of Instant-NGP, Müller et al. 2022)
    Encodes coordinates through a multiresolution hashed feature grid before a
    small MLP head.

``Siren`` (Sitzmann et al. 2020)
    MLP using ``sin`` activations with the initialisation from the original
    paper; particularly effective for high-frequency and implicit representations.

``ModifiedMLP`` (Wang et al. 2022)
    Adds U/V encoder branches that gate hidden states for richer
    coordinate-conditioned representations.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, cast

import torch
from torch import Tensor, nn

from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import StandardEntryConsumer
from dlkit.domain.nn.ffnn.constrained import FactorizedFFNN
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.primitives import (
    DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
    DEFAULT_SCALE_EQUIVARIANT_NORM,
    ScaleEquivariantWrapper,
)
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation

_DEFAULT_NORM = DEFAULT_SCALE_EQUIVARIANT_NORM
_DEFAULT_EPS_GAIN = DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN
_DEFAULT_HASH_PRIMES = (
    1,
    2_654_435_761,
    805_459_861,
    36_776_121,
    20_909_719,
    1_437_569,
    1_934_966_399,
    83_492_791,
)


def _fourier_encode(x: Tensor, B: Tensor) -> Tensor:
    """Return the standard sine/cosine Fourier feature encoding."""
    proj = 2 * math.pi * x @ B.T
    return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FourierFeatureNetwork(StandardEntryConsumer, nn.Module):
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

    class InputSpec(_InputSpec):
        pass

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        B = sigma * torch.randn(n_frequencies, in_features)
        if learnable_B:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.mlp = FFNN(
            in_features=2 * n_frequencies,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=resolve_activation(activation),
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
        return self.mlp(_fourier_encode(x, self.B))


class FactorizedFourierFeatureNetwork(StandardEntryConsumer, nn.Module):
    """Fourier-feature coordinate network with paper-style factorized MLP backbone."""

    class InputSpec(_InputSpec):
        pass

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
        mean: float = 1.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        B = sigma * torch.randn(n_frequencies, in_features)
        if learnable_B:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

        self.mlp = FactorizedFFNN(
            in_features=2 * n_frequencies,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            mean=mean,
            std=std,
            activation=resolve_activation(activation),
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(_fourier_encode(x, self.B))


class MultiresolutionHashEncoding(nn.Module):
    """Multiresolution hashed grid encoding for low-dimensional coordinates."""

    def __init__(
        self,
        *,
        in_features: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        bounds: tuple[tuple[float, float], ...] | None = None,
        include_input: bool = True,
    ) -> None:
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if log2_hashmap_size <= 0:
            raise ValueError("log2_hashmap_size must be positive")
        if base_resolution <= 0:
            raise ValueError("base_resolution must be positive")
        if finest_resolution < base_resolution:
            raise ValueError("finest_resolution must be >= base_resolution")

        super().__init__()
        self.in_features = in_features
        self.num_levels = num_levels
        self.features_per_level = features_per_level
        self.include_input = include_input
        self.hashmap_size = 1 << log2_hashmap_size

        bounds_value = bounds or tuple((-1.0, 1.0) for _ in range(in_features))
        if len(bounds_value) != in_features:
            raise ValueError("bounds must provide exactly one (min, max) pair per input feature")

        mins = []
        maxs = []
        for low, high in bounds_value:
            if high <= low:
                raise ValueError("each bounds pair must satisfy low < high")
            mins.append(low)
            maxs.append(high)

        bounds_min = torch.tensor(mins, dtype=torch.float32)
        self.register_buffer("bounds_min", bounds_min)
        self.register_buffer("bounds_range", torch.tensor(maxs, dtype=torch.float32) - bounds_min)
        self.register_buffer(
            "hash_primes",
            torch.tensor(
                [_DEFAULT_HASH_PRIMES[i % len(_DEFAULT_HASH_PRIMES)] for i in range(in_features)],
                dtype=torch.long,
            ),
        )

        if num_levels == 1:
            resolutions = [base_resolution]
        else:
            growth = math.exp(math.log(finest_resolution / base_resolution) / (num_levels - 1))
            resolutions = [
                int(math.floor(base_resolution * (growth**level))) for level in range(num_levels)
            ]
        self.resolutions = tuple(max(1, resolution) for resolution in resolutions)

        self.tables = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.hashmap_size, features_per_level))
                for _ in range(num_levels)
            ]
        )
        for table in self.tables:
            nn.init.uniform_(table, -1e-4, 1e-4)

        input_width = in_features if include_input else 0
        self.output_dim = input_width + num_levels * features_per_level

    def _normalize(self, x: Tensor) -> Tensor:
        bounds_min = cast(Tensor, self.bounds_min).to(device=x.device, dtype=x.dtype)
        bounds_range = cast(Tensor, self.bounds_range).to(device=x.device, dtype=x.dtype)
        return ((x - bounds_min) / bounds_range).clamp(0.0, 1.0)

    def _hash_indices(self, grid_indices: Tensor) -> Tensor:
        hashed = torch.zeros_like(grid_indices[..., 0], dtype=torch.long)
        hash_primes = cast(Tensor, self.hash_primes)
        for dim in range(self.in_features):
            hashed ^= grid_indices[..., dim] * hash_primes[dim]
        return hashed.remainder(self.hashmap_size)

    def _interpolate_level(self, normalized: Tensor, *, level: int) -> Tensor:
        resolution = self.resolutions[level]
        table = cast(Tensor, self.tables[level])
        scaled = normalized * resolution
        lower = torch.floor(scaled).to(dtype=torch.long).clamp(0, max(resolution - 1, 0))
        frac = (scaled - lower.to(dtype=normalized.dtype)).clamp(0.0, 1.0)

        encoded = torch.zeros(
            *normalized.shape[:-1],
            self.features_per_level,
            device=normalized.device,
            dtype=table.dtype,
        )
        for corner in range(1 << self.in_features):
            weight = torch.ones_like(frac[..., 0])
            offset_components: list[Tensor] = []
            for dim in range(self.in_features):
                bit = (corner >> dim) & 1
                if bit:
                    weight = weight * frac[..., dim]
                    offset_components.append(torch.ones_like(lower[..., dim]))
                else:
                    weight = weight * (1.0 - frac[..., dim])
                    offset_components.append(torch.zeros_like(lower[..., dim]))

            offsets = torch.stack(offset_components, dim=-1)
            hashed = self._hash_indices(lower + offsets)
            encoded = encoded + weight.unsqueeze(-1) * table[hashed]
        return encoded

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected x.shape[-1] == {self.in_features}, got {x.shape[-1]}.")
        normalized = self._normalize(x)
        levels = [
            self._interpolate_level(normalized, level=level) for level in range(self.num_levels)
        ]
        if self.include_input:
            return torch.cat([x, *levels], dim=-1)
        return torch.cat(levels, dim=-1)


class HashEncodingNetwork(StandardEntryConsumer, nn.Module):
    """Coordinate network using a multiresolution hashed grid encoder."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        num_levels: int = 16,
        features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        bounds: tuple[tuple[float, float], ...] | None = None,
        include_input: bool = True,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoding = MultiresolutionHashEncoding(
            in_features=in_features,
            num_levels=num_levels,
            features_per_level=features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            bounds=bounds,
            include_input=include_input,
        )
        self.mlp = FFNN(
            in_features=self.encoding.output_dim,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=resolve_activation(activation),
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(self.encoding(x))


class Siren(StandardEntryConsumer, nn.Module):
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

    class InputSpec(_InputSpec):
        pass

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


class ModifiedMLP(StandardEntryConsumer, nn.Module):
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
        num_layers: Number of hidden linear layers (>= 2).
        activation: Gating activation σ. Defaults to torch.sigmoid.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = "sigmoid",
    ) -> None:
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 (U/V gating requires at least one hidden layer)"
            )
        super().__init__()
        self.activation = resolve_activation(activation)
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


class ScaleEquivariantFourierFeatureNetwork(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant Fourier feature network."""

    class InputSpec(_InputSpec):
        pass

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=FourierFeatureNetwork(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_frequencies=n_frequencies,
                sigma=sigma,
                learnable_B=learnable_B,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantSiren(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant sinusoidal representation network."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        omega0: float = 30.0,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=Siren(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                omega0=omega0,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantModifiedMLP(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant modified MLP with U/V gating."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = "sigmoid",
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=ModifiedMLP(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=resolve_activation(activation),
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantFactorizedFourierFeatureNetwork(
    StandardEntryConsumer, ScaleEquivariantWrapper
):
    """Scale-equivariant factorized Fourier-feature network."""

    class InputSpec(_InputSpec):
        pass

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
        mean: float = 1.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=FactorizedFourierFeatureNetwork(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                n_frequencies=n_frequencies,
                sigma=sigma,
                learnable_B=learnable_B,
                mean=mean,
                std=std,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )
