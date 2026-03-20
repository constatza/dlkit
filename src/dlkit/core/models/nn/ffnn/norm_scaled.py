from __future__ import annotations

from typing import Literal
from collections.abc import Callable

import torch
from torch import Tensor, nn

from dlkit.core.models.nn.base import DLKitModel
from dlkit.core.models.nn.ffnn.simple import ConstantWidthFFNN
from dlkit.core.models.nn.primitives.parametrized_layers import (
    FactorizedLinear,
    SPDFactorizedLinear,
    SPDLinear,
    SymmetricFactorizedLinear,
    SymmetricLinear,
)


class NormScaledFFNN(DLKitModel):
    """Wrap a base FFNN with input/output norm scaling.

    This module enforces homogeneous scaling consistency for Ax = b by:
      1) Normalizing b: b_scaled = b / ||b||_p
      2) Predicting x_scaled = base_model(b_scaled)
      3) Rescaling x: x = x_scaled * ||b||_p

    Precision is managed by PyTorch Lightning's precision plugins via the Trainer.

    Args:
        base_model: Underlying module that operates on normalized inputs.
        norm: Which vector norm to use; one of {"l2", "l1", "linf"}.
        eps_gain: Multiplier for machine epsilon used to avoid division by zero.
        keep_stats: When True, forward returns (x, {"norm": norms}).
    """

    SUPPORTED_NORMS = {"l2", "l1", "linf"}
    DEFAULT_NORM = "l2"
    DEFAULT_EPS_GAIN = 10.0

    def __init__(
        self,
        *,
        base_model: nn.Module,
        norm: str = DEFAULT_NORM,
        eps_gain: float = DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        if not isinstance(base_model, nn.Module):
            raise TypeError("base_model must be an instance of torch.nn.Module")
        if norm not in self.SUPPORTED_NORMS:
            raise ValueError(f"norm must be one of {self.SUPPORTED_NORMS}, got {norm!r}.")
        if eps_gain <= 0:
            raise ValueError("eps_gain must be > 0.")

        super().__init__()
        self.base_model = base_model
        self.norm = norm
        self.eps_gain = float(eps_gain)
        self.keep_stats = keep_stats

    @staticmethod
    def _compute_eps(x: Tensor, gain: float) -> float:
        """Compute epsilon scaled by gain.

        Args:
            x: Input tensor (used for dtype info).
            gain: Gain multiplier for machine epsilon.

        Returns:
            Scaled epsilon value.
        """
        finfo = torch.finfo(x.dtype)
        return float(gain * finfo.eps)

    def _vector_norm(self, x: Tensor) -> Tensor:
        """Compute per-sample vector norm.

        Args:
            x: Input tensor.

        Returns:
            Vector norms with keepdim=True.
        """
        match self.norm:
            case "l2":
                return torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
            case "l1":
                return torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True)
            case _:
                return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Forward pass with norm scaling.

        Args:
            x: Input tensor (floating point).

        Returns:
            Scaled output, or tuple of (output, {"norm": norms}) if keep_stats=True.
        """
        if not torch.is_floating_point(x):
            raise TypeError(f"Expected floating point tensor, received dtype={x.dtype}.")
        if x.ndim < 1:
            raise ValueError(
                f"Expected x to have at least 1 dimension, got shape {tuple(x.shape)}."
            )

        norms = self._vector_norm(x)
        eps = self._compute_eps(x, self.eps_gain)
        safe_div = norms.clamp_min(eps)

        x_scaled = self.base_model(x / safe_div) * norms

        if self.keep_stats:
            return x_scaled, {"norm": norms}
        return x_scaled


class NormScaledLinearFFNN(NormScaledFFNN):
    """NormScaledFFNN with a single Linear layer as the base network.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include bias in the linear layer.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = True,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=nn.Linear(in_features, out_features, bias=bias),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledConstantWidthFFNN(NormScaledFFNN):
    """NormScaledFFNN backed by ConstantWidthFFNN.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        hidden_size: Size of hidden layers.
        num_layers: Number of hidden layers.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
        activation: Activation function.
        normalize: Type of normalization.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        base_model = ConstantWidthFFNN(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation if activation is not None else nn.functional.gelu,
            normalize=normalize,
            dropout=dropout,
        )
        super().__init__(
            base_model=base_model,
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledSymmetricLinear(NormScaledFFNN):
    """NormScaledFFNN backed by a SymmetricLinear layer.

    Args:
        features: Square matrix dimension.
        bias: Whether to include a bias term.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        features: int,
        bias: bool = False,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=SymmetricLinear(
                features=features,
                bias=bias,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledSPDLinear(NormScaledFFNN):
    """NormScaledFFNN backed by an SPDLinear layer.

    Args:
        features: Square matrix dimension.
        bias: Whether to include a bias term.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        features: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=SPDLinear(
                features=features,
                bias=bias,
                min_diag=min_diag,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledFactorizedLinear(NormScaledFFNN):
    """NormScaledFFNN backed by a FactorizedLinear layer.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        mean: Mean for log-scale initialisation.
        std: Standard deviation for log-scale initialisation.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=FactorizedLinear(
                in_features=in_features,
                out_features=out_features,
                bias=bias,
                mean=mean,
                std=std,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledSymmetricFactorizedLinear(NormScaledFFNN):
    """NormScaledFFNN backed by a SymmetricFactorizedLinear layer.

    Args:
        features: Square matrix dimension.
        bias: Whether to include a bias term.
        mean: Mean for log-scale initialisation.
        std: Standard deviation for log-scale initialisation.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        features: int,
        bias: bool = False,
        mean: float = 0.0,
        std: float = 0.1,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=SymmetricFactorizedLinear(
                features=features,
                bias=bias,
                mean=mean,
                std=std,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class NormScaledSPDFactorizedLinear(NormScaledFFNN):
    """NormScaledFFNN backed by an SPDFactorizedLinear layer.

    Args:
        features: Square matrix dimension.
        bias: Whether to include a bias term.
        min_diag: Positive diagonal slack floor for SPD enforcement.
        mean: Mean for log-scale initialisation.
        std: Standard deviation for log-scale initialisation.
        norm: Vector norm type to use.
        eps_gain: Epsilon gain multiplier.
        keep_stats: Whether to return norm statistics.
    """

    def __init__(
        self,
        *,
        features: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        norm: str = NormScaledFFNN.DEFAULT_NORM,
        eps_gain: float = NormScaledFFNN.DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
    ) -> None:
        super().__init__(
            base_model=SPDFactorizedLinear(
                features=features,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )
