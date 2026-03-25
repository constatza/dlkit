from collections.abc import Callable
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
    return x_in + x_out


def agg_concat(x_in: torch.Tensor, x_out: torch.Tensor) -> torch.Tensor:
    return torch.cat([x_in, x_out], dim=1)  # Concatenating along the channel dimension


aggregation_functions = {
    "sum": agg_sum,
    "concat": agg_concat,
}


def _detect_channels(module: nn.Module) -> tuple[int | None, int | None]:
    """Detect input and output channels from a module using guard clauses.

    Args:
        module (nn.Module): The module to detect channels from.

    Returns:
        tuple[int | None, int | None]: (in_channels, out_channels) or (None, None) if not found.
    """
    if hasattr(module, "in_channels") and hasattr(module, "out_channels"):
        m = cast(_HasChannels, module)
        return int(m.in_channels), int(m.out_channels)
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        m2 = cast(_HasFeatures, module)
        return int(m2.in_features), int(m2.out_features)
    return None, None


class SkipConnection(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        how: Literal["sum", "concat"] = "sum",
        layer_type: Literal["conv1d", "conv2d", "linear"] = "conv1d",
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.Identity(),
        in_channels: int | None = None,
        out_channels: int | None = None,
        kernel_size: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        """Initializes the SkipConnection.

        Args:
            module (nn.Module): The module to apply to the input.
            how (Literal["sum", "concat"], optional): Aggregation method. Defaults to "sum".
            layer_type (Literal["conv1d", "conv2d", "linear"], optional): Type of layer for adaptation. Defaults to "conv1d".
            activation (Callable, optional): Activation function. Defaults to nn.Identity().
            in_channels (int | None, optional): Input channels. Auto-detected if not provided. Defaults to None.
            out_channels (int | None, optional): Output channels. Auto-detected if not provided. Defaults to None.
            kernel_size (int, optional): Kernel size. Defaults to 1.
            stride (int, optional): Stride. Defaults to 1.
            bias (bool, optional): Whether to use bias. Defaults to True.
        """
        super().__init__()
        detected_in, detected_out = _detect_channels(module)
        self.in_channels = detected_in if detected_in is not None else in_channels
        self.out_channels = detected_out if detected_out is not None else out_channels

        if hasattr(module, "dilation"):
            self.dilation = module.dilation
        if hasattr(module, "stride"):
            self.stride = module.stride

        if self.in_channels is None or self.out_channels is None:
            raise ValueError("in_channels and out_channels must be specified")

        self.reduce_layer = select_skip_layers(
            layer_type,
            self.in_channels,
            self.out_channels,
            stride,
            bias=bias,
        )
        self.kernel_size = kernel_size
        self.activation = activation
        self.layer_type = layer_type
        self.module = module

        self.aggregation_function = aggregation_functions[how]

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x_out = self.module(x_in)
        skip = self.reduce_layer(x_in)
        agg_out = self.aggregation_function(skip, x_out)
        return self.activation(agg_out)


def select_skip_layers(
    layer_type: Literal["conv1d", "conv2d", "linear"],
    in_channels: int,
    out_channels: int,
    stride: int,
    bias: bool = True,
) -> nn.Module:
    """Select and instantiate a skip adaptation layer.

    Args:
        layer_type (Literal["conv1d", "conv2d", "linear"]): Type of adaptation layer.
        in_channels (int): Input channel count.
        out_channels (int): Output channel count.
        stride (int): Stride for the adaptation layer.
        bias (bool, optional): Whether to use bias. Defaults to True.

    Returns:
        nn.Module: The selected skip adaptation layer.

    Raises:
        ValueError: If layer_type is not recognized.
    """
    if in_channels == out_channels:
        return nn.Identity()
    match layer_type:
        case "conv1d":
            return nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=bias)
        case "conv2d":
            return nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=bias)
        case "linear":
            return nn.Linear(in_channels, out_channels, bias=False)
        case _:
            raise ValueError(f"Unsupported layer type: {layer_type!r}")
