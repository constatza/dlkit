"""Frequency-enhanced feed-forward network families.

Two composable base classes are provided for users who want to inject
custom architectures:

``FourierAugmented(backbone, n_modes)``
    Wraps *any* flat-input ``nn.Module`` with Fourier feature augmentation.
    The backbone must accept inputs of size ``original_in + n_modes * 2``.

``SpectralDualPath(spatial_branch, spectral_branch, projection, n_modes, merge)``
    Parallel spatial + spectral branches with injectable sub-networks and
    projection layer.

And two convenience constructors that build the sub-networks from scalar
parameters:

``FourierEnhancedFFNN`` — inherits ``FourierAugmented``; builds a
    ``FFNN`` backbone internally.

``DualPathFFNN`` — inherits ``SpectralDualPath``; builds two
    ``FFNN`` branches and a ``Linear`` projection internally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Self

import torch
from torch import Tensor, nn

from dlkit.domain.nn.contracts import ModelContractSpec, TabulaRSpec
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation

# ---------------------------------------------------------------------------
# Shared spectral-feature helper (pure function, no learnable parameters)
# ---------------------------------------------------------------------------


def _spectral_features(x: Tensor, n_modes: int) -> Tensor:
    """Compute zero-padded real+imag Fourier features of length ``n_modes * 2``.

    Treats the last dimension of *x* as a 1-D signal and returns the truncated
    real FFT stacked as ``[real₀…real_{m-1}, imag₀…imag_{m-1}]``.  The
    spectrum is truncated or zero-padded so the output width is always
    ``n_modes * 2``, keeping the downstream module's input dimension constant.

    **When to use:** Appropriate when input features ARE samples of a signal at
    equally-spaced indices (e.g. time-series channels, FNO-style inputs).  For
    PDE coordinate inputs such as ``(x, y, t)`` the DFT of the feature vector
    has no physical meaning — use :class:`~dlkit.domain.nn.spectral.coordinate.FourierFeatureNetwork`
    instead, which applies a coordinate-wise sinusoidal mapping γ(x) = [sin(2πBx), cos(2πBx)].

    Args:
        x: Input tensor of shape ``(batch, in_features)``.
        n_modes: Number of complex Fourier coefficients to retain.

    Returns:
        Tensor of shape ``(batch, n_modes * 2)``.
    """
    x_ft = torch.fft.rfft(x, norm="ortho")
    n_avail = x_ft.shape[-1]
    if n_avail >= n_modes:
        x_ft_out = x_ft[..., :n_modes]
    else:
        pad = torch.zeros(
            *x_ft.shape[:-1],
            n_modes - n_avail,
            dtype=x_ft.dtype,
            device=x.device,
        )
        x_ft_out = torch.cat([x_ft, pad], dim=-1)
    return torch.cat([x_ft_out.real, x_ft_out.imag], dim=-1)


# ---------------------------------------------------------------------------
# Composable base classes
# ---------------------------------------------------------------------------


class FourierAugmented(nn.Module):
    """Wrap any flat-input module with Fourier feature augmentation.

    Computes the truncated real FFT of the input, stacks real and imaginary
    parts into a vector of length ``n_modes * 2``, and concatenates it with
    the original input before forwarding through the backbone.

    The backbone must accept inputs of size
    ``original_in_features + n_modes * 2``.

    Args:
        backbone: Any ``nn.Module`` that maps
            ``(batch, in_features + n_modes * 2) → (batch, out_features)``.
        n_modes: Number of Fourier modes (complex coefficients) to retain.

    Example — inject a custom residual network::

        class MyResNet(nn.Module):
            def __init__(self): ...  # input size = 16 + 12 = 28 (in_features=16, n_modes=6)


        model = FourierAugmented(backbone=MyResNet(), n_modes=6)
        y = model(x)  # x: (B, 16)
    """

    def __init__(self, *, backbone: nn.Module, n_modes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self._n_modes = n_modes

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with spectral augmentation.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor produced by the backbone.
        """
        freq = _spectral_features(x, self._n_modes)
        return self.backbone(torch.cat([x, freq], dim=-1))


class SpectralDualPath(nn.Module):
    """Parallel spatial + spectral branches with injectable sub-networks.

    Runs a spatial branch on the raw input and a spectral branch on the
    truncated Fourier representation, then merges and projects the outputs.

    Args:
        spatial_branch: Processes the raw input tensor.
            Input: ``(batch, in_features)`` → output: ``(batch, hidden_size)``.
        spectral_branch: Processes the spectral features.
            Input: ``(batch, n_modes * 2)`` → output: ``(batch, hidden_size)``.
        projection: Maps the merged representation to the output.
            For ``merge="add"``: ``(batch, hidden_size) → (batch, out_features)``.
            For ``merge="concat"``: ``(batch, hidden_size * 2) → (batch, out_features)``.
        n_modes: Number of Fourier modes retained for the spectral branch.
        merge: ``"add"`` for element-wise sum; ``"concat"`` for concatenation.

    Example — inject transformer encoders as branches::

        spatial = TransformerEncoder(in_features=64, hidden=128)
        spectral = TransformerEncoder(in_features=12, hidden=128)
        proj = nn.Linear(128, 4)
        model = SpectralDualPath(
            spatial_branch=spatial,
            spectral_branch=spectral,
            projection=proj,
            n_modes=6,
            merge="add",
        )
    """

    def __init__(
        self,
        *,
        spatial_branch: nn.Module,
        spectral_branch: nn.Module,
        projection: nn.Module,
        n_modes: int,
        merge: Literal["add", "concat"] = "add",
    ) -> None:
        if merge not in {"add", "concat"}:
            raise ValueError(f"merge must be 'add' or 'concat', got {merge!r}")
        super().__init__()
        self.spatial_branch = spatial_branch
        self.spectral_branch = spectral_branch
        self.projection = projection
        self._n_modes = n_modes
        self._merge = merge

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through both branches.

        Args:
            x: Input tensor of shape ``(batch, in_features)``.

        Returns:
            Output tensor of shape ``(batch, out_features)``.
        """
        h_spatial = self.spatial_branch(x)
        h_spectral = self.spectral_branch(_spectral_features(x, self._n_modes))
        if self._merge == "add":
            merged = h_spatial + h_spectral
        else:
            merged = torch.cat([h_spatial, h_spectral], dim=-1)
        return self.projection(merged)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


class FourierEnhancedFFNN(FourierAugmented):
    """FFNN augmented with truncated Fourier features.

    Inherits ``FourierAugmented`` and builds a ``FFNN`` backbone
    whose input size is ``in_features + n_modes * 2``.

    The factory "ffnn" strategy injects ``in_features`` and ``out_features``
    automatically from the dataset shape summary.

    Args:
        in_features: Number of spatial input features.
        out_features: Number of output features.
        hidden_size: Width of all hidden layers in the internal MLP.
        num_layers: Number of hidden layers in the internal MLP.
        n_modes: Number of Fourier modes retained.
        activation: Pointwise activation for the internal MLP.
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
        n_modes: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        backbone = FFNN(
            in_features=in_features + n_modes * 2,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=resolve_activation(activation),
            normalize=normalize,
            dropout=dropout,
        )
        super().__init__(backbone=backbone, n_modes=n_modes)

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Build the network from a model contract spec."""
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


class DualPathFFNN(SpectralDualPath):
    """FFNN with parallel spatial and spectral ``FFNN`` branches.

    Inherits ``SpectralDualPath`` and builds both branches as
    ``FFNN`` instances.

    Args:
        in_features: Number of spatial input features.
        out_features: Number of output features.
        hidden_size: Width of the hidden layers in both branches.
        num_layers: Number of hidden layers in both branches.
        n_modes: Number of Fourier modes retained for the spectral branch.
        merge: ``"add"`` for element-wise sum; ``"concat"`` for concatenation
            followed by linear projection.
        activation: Pointwise activation for both branches.
        normalize: Normalisation type for both branches.
        dropout: Dropout probability for both branches.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        n_modes: int,
        merge: Literal["add", "concat"] = "add",
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        resolved = resolve_activation(activation)
        spatial_branch = FFNN(
            in_features=in_features,
            out_features=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
        )
        spectral_branch = FFNN(
            in_features=n_modes * 2,
            out_features=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
        )
        proj_in = hidden_size * 2 if merge == "concat" else hidden_size
        projection = nn.Linear(proj_in, out_features)
        super().__init__(
            spatial_branch=spatial_branch,
            spectral_branch=spectral_branch,
            projection=projection,
            n_modes=n_modes,
            merge=merge,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Build the network from a model contract spec."""
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )
