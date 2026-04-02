from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.ffnn.parametric import (
    ConstantWidthParametricFFNN,
    EmbeddedParametricFFNN,
)
from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    SPDFactorizedLinear,
    SPDLinear,
)


class ConstantWidthSPDFFNN(ConstantWidthParametricFFNN):
    """Constant-width network with SPD-constrained linear layers.

    Args:
        size: Square feature size (in == out).
        num_layers: Number of SPD blocks.
        residual: Whether to wrap each block in a skip connection.
        bias: Whether to include bias in each SPD layer.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        pos_fn: Element-wise positive activation for diagonal enforcement (default: softplus).
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=lambda n: SPDLinear(n, bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSPDFactorizedFFNN(ConstantWidthParametricFFNN):
    """Constant-width network with SPD-factorized linear layers (W = D @ SPD(A) @ D).

    Args:
        size: Square feature size (in == out).
        num_layers: Number of SPD-factorized blocks.
        residual: Whether to wrap each block in a skip connection.
        bias: Whether to include bias in each layer.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        mean: Mean for sandwich-scale log initialisation.
        std: Std for sandwich-scale log initialisation.
        pos_fn: Element-wise positive activation for diagonal enforcement (default: softplus).
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=lambda n: SPDFactorizedLinear(
                n, bias=bias, min_diag=min_diag, mean=mean, std=std, pos_fn=pos_fn
            ),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthFactorizedFFNN(ConstantWidthParametricFFNN):
    """Constant-width network with row-scale factorized linear layers (W = diag(pos_fn(s)) @ A).

    Args:
        size: Square feature size (in == out).
        num_layers: Number of factorized blocks.
        residual: Whether to wrap each block in a skip connection.
        bias: Whether to include bias in each layer.
        mean: Mean for log-scale initialisation.
        std: Std for log-scale initialisation.
        pos_fn: Element-wise function mapping scale to positive values (default: exp).
        activation: Activation function (default: gelu).
        normalize: Normalization type ('batch', 'layer', or None).
        dropout: Dropout probability (default: 0.0).
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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=lambda n: FactorizedLinear(
                n, n, bias=bias, mean=mean, std=std, pos_fn=pos_fn
            ),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSPDFFNN(EmbeddedParametricFFNN):
    """Embedded-parametric network with SPD-constrained body layers.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of SPD blocks.
        residual: Whether to use skip connections in the body.
        bias: Whether to include bias in each SPD layer.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        pos_fn: Element-wise positive activation for diagonal enforcement (default: softplus).
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
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=lambda n: SPDLinear(n, bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSPDFactorizedFFNN(EmbeddedParametricFFNN):
    """Embedded-parametric network with SPD-factorized body layers (W = D @ SPD(A) @ D).

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of SPD-factorized blocks.
        residual: Whether to use skip connections in the body.
        bias: Whether to include bias in each layer.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        mean: Mean for sandwich-scale log initialisation.
        std: Std for sandwich-scale log initialisation.
        pos_fn: Element-wise positive activation for diagonal enforcement (default: softplus).
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
        residual: bool = False,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=lambda n: SPDFactorizedLinear(
                n, bias=bias, min_diag=min_diag, mean=mean, std=std, pos_fn=pos_fn
            ),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedFactorizedFFNN(EmbeddedParametricFFNN):
    """Embedded-parametric network with row-scale factorized body layers.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Square size of the parametric body.
        num_layers: Number of factorized blocks.
        residual: Whether to use skip connections in the body.
        bias: Whether to include bias in each layer.
        mean: Mean for log-scale initialisation.
        std: Std for log-scale initialisation.
        pos_fn: Element-wise function mapping scale to positive values (default: exp).
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
        residual: bool = False,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=lambda n: FactorizedLinear(
                n, n, bias=bias, mean=mean, std=std, pos_fn=pos_fn
            ),
            residual=residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
