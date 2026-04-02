from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.ffnn.norm_scaled import NormScaledFFNN
from dlkit.domain.nn.ffnn.parametric_variants import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
)


class NormScaledConstantWidthSPDFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by ConstantWidthSPDFFNN.

    Args:
        size: Square feature size (in == out).
        num_layers: Number of SPD blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in SPD layers.
        min_diag: Positive diagonal slack floor.
        pos_fn: Positive activation for SPD diagonal enforcement.
        norm: Vector norm type ('l2', 'l1', 'linf').
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSPDFFNN(
                size=size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledConstantWidthSPDFactorizedFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by ConstantWidthSPDFactorizedFFNN.

    Args:
        size: Square feature size (in == out).
        num_layers: Number of SPD-factorized blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in layers.
        min_diag: Positive diagonal slack floor.
        mean: Mean for sandwich-scale log initialisation.
        std: Std for sandwich-scale log initialisation.
        pos_fn: Positive activation for SPD diagonal enforcement.
        norm: Vector norm type.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSPDFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledConstantWidthFactorizedFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by ConstantWidthFactorizedFFNN.

    Args:
        size: Square feature size (in == out).
        num_layers: Number of factorized blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in layers.
        mean: Mean for log-scale initialisation.
        std: Std for log-scale initialisation.
        pos_fn: Positive activation for scale (default: exp).
        norm: Vector norm type.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledEmbeddedSPDFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by EmbeddedSPDFFNN.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of SPD blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in SPD layers.
        min_diag: Positive diagonal slack floor.
        pos_fn: Positive activation for SPD diagonal enforcement.
        norm: Vector norm type.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSPDFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledEmbeddedSPDFactorizedFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by EmbeddedSPDFactorizedFFNN.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of SPD-factorized blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in layers.
        min_diag: Positive diagonal slack floor.
        mean: Mean for sandwich-scale log initialisation.
        std: Std for sandwich-scale log initialisation.
        pos_fn: Positive activation for SPD diagonal enforcement.
        norm: Vector norm type.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSPDFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledEmbeddedFactorizedFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by EmbeddedFactorizedFFNN.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of factorized blocks.
        residual: Whether to use skip connections.
        bias: Whether to include bias in layers.
        mean: Mean for log-scale initialisation.
        std: Std for log-scale initialisation.
        pos_fn: Positive activation for scale (default: exp).
        norm: Vector norm type.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Normalization type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        residual: bool = False,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                residual=residual,
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=activation or nn.functional.gelu,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )
