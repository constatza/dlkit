from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _FactorizedLinearBase(nn.Module):
    """Shared implementation for row-wise factorized linear layers.

    Stores two independent parameters — ``base_weight`` and ``log_scale`` —
    and computes the effective weight as ``W = diag(phi(log_scale)) @ base_weight``.

    Using a plain :class:`~torch.nn.Module` (rather than ``parametrize``)
    keeps the state dict flat and the semantics clear: the factorisation is a
    *modelling choice*, not a manifold constraint.

    Attributes:
        base_weight (nn.Parameter): Raw weight of shape ``(out_features, in_features)``.
        log_scale (nn.Parameter): Learnable log-scale of shape ``(out_features,)``.
        bias (nn.Parameter | None): Optional bias of shape ``(out_features,)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mean: float,
        std: float,
        pos_fn: Callable[[Tensor], Tensor],
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the factorized linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size.
            bias: Whether to include a bias term.
            mean: Mean of the Gaussian used to sample ``log_scale``.
            std: Standard deviation for log-scale initialisation.
            pos_fn: Element-wise function mapping log-scale to positive scale
                factors.
            device: Optional device for parameter initialisation.
            dtype: Optional dtype for parameter initialisation.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._log_scale_mean = float(mean)
        self._log_scale_std = float(std)
        self._pos_fn = pos_fn
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.log_scale = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters for the factorized linear layer."""
        nn.init.kaiming_uniform_(self.base_weight, a=0.0)
        nn.init.normal_(self.log_scale, mean=self._log_scale_mean, std=self._log_scale_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight ``diag(pos_fn(log_scale)) @ base_weight``."""
        return self._pos_fn(self.log_scale).unsqueeze(1) * self.base_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factorized linear transformation.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, out_features)``.
        """
        return F.linear(x, self.weight, self.bias)


class FactorizedLinear(_FactorizedLinearBase):
    """Paper-style random-weight-factorized linear layer.

    This is the public rectangular factorized primitive. It fixes
    ``phi = exp`` and interprets ``mean`` / ``std`` as the literal Gaussian
    parameters used to sample the latent scale variable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mean: float = 0.0,
        std: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the public factorized linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size.
            bias: Whether to include a bias term.
            mean: Gaussian mean for ``log_scale`` initialisation
                (``0.0`` -> ``exp(0) = 1.0``, unit scale at init).
            std: Standard deviation for log-scale initialisation.
            device: Optional device for parameter initialisation.
            dtype: Optional dtype for parameter initialisation.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mean=mean,
            std=std,
            pos_fn=torch.exp,
            device=device,
            dtype=dtype,
        )


class SoftplusFactorizedLinear(_FactorizedLinearBase):
    """Advanced rectangular factorized linear layer with softplus row scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        mean: float = 0.0,
        std: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mean=mean,
            std=std,
            pos_fn=F.softplus,
            device=device,
            dtype=dtype,
        )
