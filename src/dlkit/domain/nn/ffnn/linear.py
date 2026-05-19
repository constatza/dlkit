from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.primitives.parametrized_layers import (
    FactorizedLinear,
    SPDFactorizedLinear,
    SPDLinear,
    SymmetricFactorizedLinear,
    SymmetricLinear,
)
from dlkit.domain.nn.utils import make_norm_layer

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


class LinearNetwork(nn.Module):
    """A simple linear network with a single layer and optional normalization.

    This network consists of a single linear transformation with optional
    batch normalization or layer normalization.

    Args:
        in_features: Size of the input features.
        out_features: Size of the output features.
        normalize: Type of normalization to apply ('batch', 'layer', or None).
        bias: Whether to include bias in the linear layer.
    """

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

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> LinearNetwork:
        """Build the network from a dataset-derived flat shape summary."""
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

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


class FactorizedLinearNetwork(nn.Module):
    """Single-layer network backed by one FactorizedLinear layer.

    The effective weight is ``W = diag(pos_fn(log_scale)) @ base_weight``,
    giving per-output-neuron scale control independent of base weight
    initialisation. No normalization wrapper — contrast with LinearNetwork.

    Args:
        in_features: Size of the input features.
        out_features: Size of the output features.
        bias: Whether to include a bias term.
        mean: Mean for log-scale initialisation (0.0 → unit scale at init).
        std: Standard deviation for log-scale initialisation.
        pos_fn: Element-wise function mapping log-scale to positive scale factors.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
    ) -> None:
        super().__init__()
        self.linear = FactorizedLinear(
            in_features, out_features, bias, mean=mean, std=std, pos_fn=pos_fn
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> FactorizedLinearNetwork:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary with in_features and out_features.
            **kwargs: Forwarded to the constructor.

        Returns:
            A new FactorizedLinearNetwork instance.
        """
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_features).

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        return self.linear(x)


class SymmetricLinearNetwork(nn.Module):
    """Single-layer network backed by one SymmetricLinear layer.

    The weight is constrained to be symmetric (W = Wᵀ). Requires
    in_features == out_features.

    Args:
        in_features: Input (and output) dimension.
        out_features: Output (and input) dimension; must equal in_features.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"SymmetricLinearNetwork requires in_features == out_features, "
                f"got {in_features} != {out_features}"
            )
        super().__init__()
        self.linear = SymmetricLinear(in_features, bias)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> SymmetricLinearNetwork:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary; in_features must equal out_features.
            **kwargs: Forwarded to the constructor.

        Returns:
            A new SymmetricLinearNetwork instance.
        """
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Output tensor of shape (batch_size, features).
        """
        return self.linear(x)


class SPDLinearNetwork(nn.Module):
    """Single-layer network backed by one SPDLinear layer.

    The weight is constrained to be symmetric positive-definite. Requires
    in_features == out_features.

    Args:
        in_features: Input (and output) dimension.
        out_features: Output (and input) dimension; must equal in_features.
        bias: Whether to include a bias term.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        pos_fn: Element-wise positive activation for diagonal enforcement.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"SPDLinearNetwork requires in_features == out_features, "
                f"got {in_features} != {out_features}"
            )
        super().__init__()
        self.linear = SPDLinear(in_features, bias, min_diag=min_diag, pos_fn=pos_fn)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> SPDLinearNetwork:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary; in_features must equal out_features.
            **kwargs: Forwarded to the constructor.

        Returns:
            A new SPDLinearNetwork instance.
        """
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Output tensor of shape (batch_size, features).
        """
        return self.linear(x)


class SymmetricFactorizedLinearNetwork(nn.Module):
    """Single-layer network backed by one SymmetricFactorizedLinear layer.

    Enforces W = D·Sym(A)·D where D = diag(exp(s)), keeping the weight
    symmetric. Requires in_features == out_features.

    Args:
        in_features: Input (and output) dimension.
        out_features: Output (and input) dimension; must equal in_features.
        bias: Whether to include a bias term.
        mean: Mean for log-scale initialisation (0.0 → unit scale at init).
        std: Standard deviation for log-scale initialisation.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mean: float = 0.0,
        std: float = 0.1,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"SymmetricFactorizedLinearNetwork requires in_features == out_features, "
                f"got {in_features} != {out_features}"
            )
        super().__init__()
        self.linear = SymmetricFactorizedLinear(in_features, bias, mean=mean, std=std)

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> SymmetricFactorizedLinearNetwork:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary; in_features must equal out_features.
            **kwargs: Forwarded to the constructor.

        Returns:
            A new SymmetricFactorizedLinearNetwork instance.
        """
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Output tensor of shape (batch_size, features).
        """
        return self.linear(x)


class SPDFactorizedLinearNetwork(nn.Module):
    """Single-layer network backed by one SPDFactorizedLinear layer.

    Enforces W = D·SPD(A)·D where D = diag(exp(s)), keeping the weight
    symmetric positive-definite. Requires in_features == out_features.

    Args:
        in_features: Input (and output) dimension.
        out_features: Output (and input) dimension; must equal in_features.
        bias: Whether to include a bias term.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        mean: Mean for log-scale initialisation (0.0 → unit scale at init).
        std: Standard deviation for log-scale initialisation.
        pos_fn: Element-wise positive activation for diagonal enforcement.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"SPDFactorizedLinearNetwork requires in_features == out_features, "
                f"got {in_features} != {out_features}"
            )
        super().__init__()
        self.linear = SPDFactorizedLinear(
            in_features, bias, min_diag=min_diag, mean=mean, std=std, pos_fn=pos_fn
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> SPDFactorizedLinearNetwork:
        """Build from a dataset-derived shape summary.

        Args:
            shape: Shape summary; in_features must equal out_features.
            **kwargs: Forwarded to the constructor.

        Returns:
            A new SPDFactorizedLinearNetwork instance.
        """
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, features).

        Returns:
            Output tensor of shape (batch_size, features).
        """
        return self.linear(x)
