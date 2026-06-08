from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import make_norm_layer, resolve_activation


@runtime_checkable
class IGatingMechanism(Protocol):
    """Protocol for gating mechanisms that combine a hidden state with optional context.

    Implementations must accept both a primary hidden tensor ``h`` and a context
    tensor ``x``, though some gates may ignore ``x`` entirely.
    """

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply the gating mechanism.

        Args:
            h (Tensor): Primary hidden-state tensor.
            x (Tensor): Context tensor (may be ignored by some implementations).

        Returns:
            Tensor: Gated output with the same leading shape as ``h``.
        """
        ...


class GLUGate(nn.Module):
    """Gated Linear Unit (Dauphin et al. 2017).

    Applies ``content_proj(h) ⊙ σ(gate_proj(h))``.  The context ``x`` is
    ignored; it is accepted only to satisfy :class:`IGatingMechanism`.

    Args:
        hidden_size (int): Dimensionality of the input and output.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialise GLUGate.

        Args:
            hidden_size (int): Size of the hidden dimension.
        """
        super().__init__()
        self.content_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: Tensor, _x: Tensor) -> Tensor:
        """Apply GLU gating.

        Args:
            h (Tensor): Primary input tensor of shape ``(..., hidden_size)``.
            _x (Tensor): Unused; accepted to satisfy IGatingMechanism protocol.

        Returns:
            Tensor: Gated output of shape ``(..., hidden_size)``.
        """
        return self.content_proj(h) * torch.sigmoid(self.gate_proj(h))


class SwiGLUGate(nn.Module):
    """Swish-Gated Linear Unit (Shazeer 2020; used in LLaMA / PaLM).

    Applies ``content_proj(h) ⊙ silu(gate_proj(h))``.  The context ``x`` is
    ignored; it is accepted only to satisfy :class:`IGatingMechanism`.

    Args:
        hidden_size (int): Dimensionality of the input and output.
    """

    def __init__(self, hidden_size: int) -> None:
        """Initialise SwiGLUGate.

        Args:
            hidden_size (int): Size of the hidden dimension.
        """
        super().__init__()
        self.content_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: Tensor, _x: Tensor) -> Tensor:
        """Apply SwiGLU gating.

        Args:
            h (Tensor): Primary input tensor of shape ``(..., hidden_size)``.
            _x (Tensor): Unused; accepted to satisfy IGatingMechanism protocol.

        Returns:
            Tensor: Gated output of shape ``(..., hidden_size)``.
        """
        return self.content_proj(h) * F.silu(self.gate_proj(h))


class GRNGate(nn.Module):
    """Gated Residual Network gate (Lim et al. 2021, Temporal Fusion Transformer).

    Architecture::

        eta1 = ELU(W1(h) + context_proj(x))    # bias absorbed into W1
        eta2 = W3(eta1)
        glu_out = eta2 ⊙ σ(W4(eta1))
        output  = LayerNorm(h + dropout(glu_out))

    When ``context_size`` is ``None``, ``context_proj`` maps
    ``hidden_size → hidden_size`` and ``x`` is expected to have the same
    width as ``h``.  When ``context_size`` is provided, ``context_proj`` maps
    ``context_size → hidden_size``.

    Args:
        hidden_size (int): Dimensionality of ``h`` and the output.
        context_size (int | None, optional): Dimensionality of ``x``.  If
            ``None``, ``x`` is assumed to have ``hidden_size`` features.
            Defaults to ``None``.
        dropout (float, optional): Dropout probability applied to the GLU
            output before the residual add.  Defaults to ``0.0``.
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialise GRNGate.

        Args:
            hidden_size (int): Size of the hidden dimension.
            context_size (int | None, optional): Size of the context dimension.
                Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super().__init__()
        ctx_in = context_size if context_size is not None else hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.context_proj = nn.Linear(ctx_in, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, hidden_size)
        self.w4 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply GRN gating with optional context.

        Args:
            h (Tensor): Primary hidden tensor of shape ``(..., hidden_size)``.
            x (Tensor): Context tensor.  Shape must be ``(..., context_size)``
                when ``context_size`` was provided, otherwise ``(..., hidden_size)``.

        Returns:
            Tensor: Output tensor of shape ``(..., hidden_size)``.
        """
        eta1 = F.elu(self.w1(h) + self.context_proj(x))
        eta2 = self.w3(eta1)
        glu_out = eta2 * torch.sigmoid(self.w4(eta1))
        return self.norm(h + self.dropout(glu_out))


class UVGate(nn.Module):
    """UV-Gate generalisation (Wang et al. 2022).

    Computes::

        z = activation(gate(h))
        U = activation(encoder_u(x))
        V = activation(encoder_v(x))
        output = z ⊙ U + (1 − z) ⊙ V

    Args:
        in_features (int): Dimensionality of the context input ``x``.
        hidden_size (int): Dimensionality of the hidden state ``h`` and output.
        activation (ActivationName | Callable[[Tensor], Tensor] | None, optional):
            Element-wise activation applied to each branch.  Defaults to ``"sigmoid"``.
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = "sigmoid",
    ) -> None:
        """Initialise UVGate.

        Args:
            in_features (int): Size of the context input dimension.
            hidden_size (int): Size of the hidden state and output dimension.
            activation (ActivationName | Callable[[Tensor], Tensor] | None, optional):
                Activation function for all three branches. Defaults to ``"sigmoid"``.
        """
        super().__init__()
        self.encoder_u = nn.Linear(in_features, hidden_size)
        self.encoder_v = nn.Linear(in_features, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.activation = resolve_activation(activation)

    def forward(self, h: Tensor, x: Tensor) -> Tensor:
        """Apply UV gating.

        Args:
            h (Tensor): Hidden state tensor of shape ``(..., hidden_size)``.
            x (Tensor): Context tensor of shape ``(..., in_features)``.

        Returns:
            Tensor: Gated output of shape ``(..., hidden_size)``.
        """
        z = self.activation(self.gate(h))
        u = self.activation(self.encoder_u(x))
        v = self.activation(self.encoder_v(x))
        return z * u + (1 - z) * v


class GatedConvolutionBlock1d(nn.Module):
    """1-D gated convolutional block.

    Applies ``Norm → Conv1d(in → 2·out) → split → content ⊙ σ(gate) → Dropout``.

    The double-width convolution produces two feature maps that are split along
    the channel dimension; the first half is the content branch and the second
    half is the sigmoid gate.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        in_timesteps (int): Sequence length (used for LayerNorm shape).
        kernel_size (int, optional): Convolution kernel size. Defaults to 3.
        stride (int, optional): Convolution stride. Defaults to 1.
        padding (str | int, optional): Padding mode or amount. Defaults to
            ``"same"``.
        normalize (NormalizerName | None, optional): Normalisation to apply
            before the convolution. Defaults to ``None``.
        dropout (float, optional): Dropout probability after gating. Defaults
            to ``0.0``.
        dilation (int, optional): Dilation rate. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution.
            Defaults to 1.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: str | int = "same",
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """Initialise GatedConvolutionBlock1d.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_timesteps (int): Sequence length for LayerNorm.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            stride (int, optional): Stride. Defaults to 1.
            padding (str | int, optional): Padding. Defaults to ``"same"``.
            normalize (NormalizerName | None, optional): Normalisation type.
                Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            dilation (int, optional): Dilation rate. Defaults to 1.
            groups (int, optional): Grouped convolution groups. Defaults to 1.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.norm = make_norm_layer(normalize, in_channels, in_timesteps)
        self.conv = nn.Conv1d(
            in_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated convolution.

        Args:
            x (Tensor): Input tensor of shape ``(B, in_channels, T)``.

        Returns:
            Tensor: Output tensor of shape ``(B, out_channels, T')``.
        """
        x = self.norm(x)
        x = self.conv(x)
        content, gate = x.chunk(2, dim=1)
        x = content * torch.sigmoid(gate)
        return self.dropout(x)


class GatedDeconvolutionBlock1d(nn.Module):
    """1-D gated transposed convolutional block.

    Applies ``Norm → ConvTranspose1d(in → 2·out) → split → content ⊙ σ(gate) → Dropout``.

    Raises:
        ValueError: If ``padding="same"`` and ``stride != 1``.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        in_timesteps (int): Sequence length (used for LayerNorm shape).
        kernel_size (int, optional): Kernel size. Defaults to 3.
        dilation (int, optional): Dilation rate. Defaults to 1.
        stride (int, optional): Stride. Defaults to 1.
        padding (str | int, optional): Padding mode or amount. ``"same"``
            maps to ``(kernel_size - 1) // 2 * dilation`` and is only valid
            when ``stride=1``. Defaults to ``"same"``.
        output_padding (int, optional): Extra size added to one output side.
            Defaults to 0.
        groups (int, optional): Grouped convolution groups. Defaults to 1.
        normalize (NormalizerName | None, optional): Normalisation type.
            Defaults to ``None``.
        dropout (float, optional): Dropout probability. Defaults to ``0.0``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        in_timesteps: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        padding: str | int = "same",
        output_padding: int = 0,
        groups: int = 1,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        """Initialise GatedDeconvolutionBlock1d.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            in_timesteps (int): Sequence length for LayerNorm.
            kernel_size (int, optional): Kernel size. Defaults to 3.
            dilation (int, optional): Dilation rate. Defaults to 1.
            stride (int, optional): Stride. Defaults to 1.
            padding (str | int, optional): Padding. Defaults to ``"same"``.
            output_padding (int, optional): Extra output size. Defaults to 0.
            groups (int, optional): Grouped convolution groups. Defaults to 1.
            normalize (NormalizerName | None, optional): Normalisation type.
                Defaults to None.
            dropout (float, optional): Dropout probability. Defaults to 0.0.

        Raises:
            ValueError: If ``padding="same"`` and ``stride != 1``.
        """
        super().__init__()
        if padding == "same" and stride != 1:
            raise ValueError(
                '"same" padding for GatedDeconvolutionBlock1d is only supported when stride=1.'
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        effective_padding: int = (
            (kernel_size - 1) // 2 * dilation if padding == "same" else int(padding)
        )
        self.norm = make_norm_layer(normalize, in_channels, in_timesteps)
        self.conv = nn.ConvTranspose1d(
            in_channels,
            2 * out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=effective_padding,
            dilation=dilation,
            groups=groups,
            output_padding=output_padding,
        )
        self.dropout = nn.Dropout1d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply gated transposed convolution.

        Args:
            x (Tensor): Input tensor of shape ``(B, in_channels, T)``.

        Returns:
            Tensor: Output tensor of shape ``(B, out_channels, T')``.
        """
        x = self.norm(x)
        x = self.conv(x)
        content, gate = x.chunk(2, dim=1)
        x = content * torch.sigmoid(gate)
        return self.dropout(x)
