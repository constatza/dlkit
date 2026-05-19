from typing import Literal, Protocol, cast

import torch
from torch import nn


class _HasChannels(Protocol):
    in_channels: int
    out_channels: int


class _HasFeatures(Protocol):
    in_features: int
    out_features: int


def agg_sum(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of skip and module outputs."""
    return x_in + x_out


def agg_concat(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    """Channel-dimension concatenation of skip and module outputs."""
    return torch.cat([x_in, x_out], dim=1)


def _detect_channels(module: nn.Module) -> tuple[int | None, int | None]:
    """Detect (in, out) channel counts from a module's attributes.

    Args:
        module (nn.Module): Module to inspect.

    Returns:
        tuple[int | None, int | None]: Detected (in, out) or (None, None).
    """
    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        m = cast(_HasChannels, module)
        return int(m.in_channels), int(m.out_channels)
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        m2 = cast(_HasFeatures, module)
        return int(m2.in_features), int(m2.out_features)
    return None, None


def _require_channels(module: nn.Module) -> tuple[int, int]:
    """Detect channels from module or raise ValueError.

    Args:
        module (nn.Module): Module to inspect.

    Returns:
        tuple[int, int]: (in_channels, out_channels).

    Raises:
        ValueError: If channel attributes cannot be found on the module.
    """
    in_ch, out_ch = _detect_channels(module)
    if in_ch is None or out_ch is None:
        raise ValueError(
            f"Cannot detect in/out channels from {type(module).__name__}. "
            "Ensure the module exposes in_channels/out_channels or in_features/out_features."
        )
    return in_ch, out_ch


def build_linear_skip_layer(module: nn.Module, *, bias: bool = True) -> nn.Module:
    """Build a linear skip adapter for a feature-based module.

    Returns ``nn.Identity`` when in==out, else ``nn.Linear(in, out, bias=bias)``.

    Args:
        module (nn.Module): Module with detectable in_features/out_features.
        bias (bool): Whether the projection layer uses a bias term.

    Returns:
        nn.Module: Identity or linear projection.

    Raises:
        ValueError: If the module has no detectable channel attributes.
    """
    in_f, out_f = _require_channels(module)
    if in_f == out_f:
        return nn.Identity()
    return nn.Linear(in_f, out_f, bias=bias)


def build_conv1d_skip_layer(
    module: nn.Module,
    *,
    stride: int = 1,
    bias: bool = True,
) -> nn.Module:
    """Build a Conv1d skip adapter for a 1D convolutional module.

    Returns ``nn.Identity`` when in==out and stride==1,
    else ``nn.Conv1d(in, out, 1, stride=stride, bias=bias)``.

    Args:
        module (nn.Module): Module with detectable in_channels/out_channels.
        stride (int): Stride for spatial downsampling in the skip path.
        bias (bool): Whether the projection layer uses a bias term.

    Returns:
        nn.Module: Identity or 1x1 convolutional projection.

    Raises:
        ValueError: If the module has no detectable channel attributes.
    """
    in_ch, out_ch = _require_channels(module)
    if in_ch == out_ch and stride == 1:
        return nn.Identity()
    return nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=bias)


def build_conv2d_skip_layer(
    module: nn.Module,
    *,
    stride: int = 1,
    bias: bool = True,
) -> nn.Module:
    """Build a Conv2d skip adapter for a 2D convolutional module.

    Returns ``nn.Identity`` when in==out and stride==1,
    else ``nn.Conv2d(in, out, 1, stride=stride, bias=bias)``.

    Args:
        module (nn.Module): Module with detectable in_channels/out_channels.
        stride (int): Stride for spatial downsampling in the skip path.
        bias (bool): Whether the projection layer uses a bias term.

    Returns:
        nn.Module: Identity or 1x1 convolutional projection.

    Raises:
        ValueError: If the module has no detectable channel attributes.
    """
    in_ch, out_ch = _require_channels(module)
    if in_ch == out_ch and stride == 1:
        return nn.Identity()
    return nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=bias)


class SkipConnection(nn.Module):
    """Residual connection wrapper implementing ``y = module(x) + skip_layer(x)``.

    The skip_layer guarantees at least a linear (or identity) path through the
    network, independent of the main module's nonlinearity.

    Use the factory functions :func:`build_linear_skip_layer`,
    :func:`build_conv1d_skip_layer`, or :func:`build_conv2d_skip_layer` to
    build the skip adapter, or inject any ``nn.Module`` directly (e.g.
    ``torch_geometric.nn.Linear`` for graph networks).

    Args:
        module (nn.Module): The main transformation module.
        skip_layer (nn.Module): The skip path module.
        how (Literal["sum", "concat"]): Aggregation mode. ``"sum"`` adds the
            paths; ``"concat"`` concatenates along dim=1, producing
            ``2 x out_channels`` total width.
    """

    def __init__(
        self,
        module: nn.Module,
        skip_layer: nn.Module,
        how: Literal["sum", "concat"] = "sum",
    ) -> None:
        super().__init__()
        if how not in ("sum", "concat"):
            raise ValueError(f"Unknown aggregation {how!r}. Expected 'sum' or 'concat'.")
        detected_in, detected_out = _detect_channels(module)
        self.in_channels = detected_in
        self.out_channels = detected_out
        self._how = how
        self.module = module
        self.reduce_layer = skip_layer

    @property
    def effective_out_channels(self) -> int:
        """Output channel count after aggregation.

        Returns:
            int: ``out_channels`` in sum mode, ``2 x out_channels`` in concat mode.

        Raises:
            ValueError: If out_channels cannot be determined from the wrapped module.
        """
        if self.out_channels is None:
            raise ValueError(
                "Cannot determine effective_out_channels: "
                f"{type(self.module).__name__} has no detectable channel count."
            )
        if self._how == "concat":
            return 2 * self.out_channels
        return self.out_channels

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """Apply module and aggregate with the skip path.

        Args:
            x_in (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Aggregated output.
        """
        x_out = self.module(x_in)
        skip = self.reduce_layer(x_in)
        if self._how == "concat":
            return agg_concat(skip, x_out)
        return agg_sum(skip, x_out)
