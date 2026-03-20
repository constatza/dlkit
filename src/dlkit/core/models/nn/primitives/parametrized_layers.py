from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parametrize

from dlkit.core.models.nn.primitives.parametrizations import (
    PositiveSandwichScale,
    SPD,
    Symmetric,
)


# ---------------------------------------------------------------------------
# Low-level registration helpers
# ---------------------------------------------------------------------------


def _register_parametrizations(
    module: nn.Module,
    tensor_name: str,
    parametrizations: Iterable[nn.Module],
    *,
    unsafe: bool = False,
) -> nn.Module:
    """Register multiple parametrizations on a module tensor.

    Args:
        module: Target module.
        tensor_name: Name of the tensor to parametrize (e.g. ``"weight"``).
        parametrizations: Parametrization modules in application order.
        unsafe: Forwarded to :func:`torch.nn.utils.parametrize.register_parametrization`.

    Returns:
        The same *module*, mutated in place.
    """
    for p in parametrizations:
        parametrize.register_parametrization(module, tensor_name, p, unsafe=unsafe)
    return module


# ---------------------------------------------------------------------------
# Public registration helpers (structural constraints only)
# ---------------------------------------------------------------------------


def register_symmetric(module: nn.Module, tensor_name: str = "weight") -> nn.Module:
    """Register hard symmetry on a module tensor.

    Args:
        module: Target module.
        tensor_name: Name of the tensor to parametrize.

    Returns:
        The same *module*, mutated in place.
    """
    return _register_parametrizations(module, tensor_name, [Symmetric()])


def register_spd(
    module: nn.Module,
    tensor_name: str = "weight",
    *,
    min_diag: float = 1e-4,
    pos_fn: Callable[[Tensor], Tensor] = F.softplus,
) -> nn.Module:
    """Register hard SPD structure on a module tensor.

    Args:
        module: Target module.
        tensor_name: Name of the tensor to parametrize.
        min_diag: Positive diagonal floor forwarded to :class:`SPD`.
        pos_fn: Element-wise positive activation used in the SPD diagonal
            enforcement (default: softplus).

    Returns:
        The same *module*, mutated in place.
    """
    return _register_parametrizations(
        module,
        tensor_name,
        [Symmetric(), SPD(min_diag=min_diag, pos_fn=pos_fn)],
    )


def register_symmetric_factorized(
    module: nn.Module,
    size: int,
    tensor_name: str = "weight",
    *,
    mean: float = 0.0,
    std: float = 0.1,
) -> nn.Module:
    """Register symmetry followed by a symmetry-preserving sandwich factorization.

    The chained parametrization enforces ``W = D @ Sym(A) @ D`` where
    ``D = diag(exp(s))``, which keeps the weight symmetric.

    Args:
        module: Target module.
        size: Square matrix dimension.
        tensor_name: Name of the tensor to parametrize.
        mean: Mean for log-scale initialisation (``0.0`` → unit scale).
        std: Standard deviation for log-scale initialisation.

    Returns:
        The same *module*, mutated in place.
    """
    return _register_parametrizations(
        module,
        tensor_name,
        [
            Symmetric(),
            PositiveSandwichScale(size=size, mean=mean, std=std),
        ],
    )


def register_spd_factorized(
    module: nn.Module,
    size: int,
    tensor_name: str = "weight",
    *,
    min_diag: float = 1e-4,
    mean: float = 0.0,
    std: float = 0.1,
    pos_fn: Callable[[Tensor], Tensor] = F.softplus,
) -> nn.Module:
    """Register SPD structure followed by an SPD-preserving sandwich factorization.

    The chained parametrization enforces ``W = D @ SPD(A) @ D``, which keeps
    the weight symmetric positive-definite.

    Args:
        module: Target module.
        size: Square matrix dimension.
        tensor_name: Name of the tensor to parametrize.
        min_diag: Positive diagonal floor forwarded to :class:`SPD`.
        mean: Mean for log-scale initialisation (``0.0`` → unit scale).
        std: Standard deviation for log-scale initialisation.
        pos_fn: Element-wise positive activation used in the SPD diagonal
            enforcement (default: softplus).

    Returns:
        The same *module*, mutated in place.
    """
    return _register_parametrizations(
        module,
        tensor_name,
        [
            Symmetric(),
            SPD(min_diag=min_diag, pos_fn=pos_fn),
            PositiveSandwichScale(size=size, mean=mean, std=std),
        ],
    )


# ---------------------------------------------------------------------------
# Parametrized linear layer classes
# ---------------------------------------------------------------------------




class SymmetricLinear(nn.Linear):
    """Linear layer whose weight is constrained to be symmetric.

    Uses :func:`register_symmetric` to enforce ``W = W^T`` via PyTorch's
    parametrize machinery.  Input and output sizes are identical.
    """

    def __init__(
        self,
        features: int,
        bias: bool = False,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the symmetric linear layer.

        Args:
            features: Square matrix dimension.
            bias: Whether to include a bias term.
            device: Optional device for weight initialisation.
            dtype: Optional dtype for weight initialisation.
        """
        super().__init__(
            in_features=features,
            out_features=features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        register_symmetric(self)


class SPDLinear(nn.Linear):
    """Linear layer whose weight is constrained to be symmetric positive-definite.

    Uses :func:`register_spd` to enforce positive-definiteness via PyTorch's
    parametrize machinery.  Input and output sizes are identical.
    """

    def __init__(
        self,
        features: int,
        bias: bool = False,
        *,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the SPD linear layer.

        Args:
            features: Square matrix dimension.
            bias: Whether to include a bias term.
            min_diag: Positive diagonal slack floor for SPD enforcement.
            pos_fn: Element-wise positive activation for diagonal enforcement
                (default: softplus).
            device: Optional device for weight initialisation.
            dtype: Optional dtype for weight initialisation.
        """
        super().__init__(
            in_features=features,
            out_features=features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        register_spd(self, min_diag=min_diag, pos_fn=pos_fn)


class FactorizedLinear(nn.Module):
    """Linear layer with an explicit positive row-wise scale factor.

    Stores two independent parameters — ``base_weight`` and ``log_scale`` —
    and computes the effective weight as
    ``W = diag(exp(log_scale)) @ base_weight``.

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
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the factorized linear layer.

        Args:
            in_features: Input feature size.
            out_features: Output feature size.
            bias: Whether to include a bias term.
            mean: Mean for log-scale initialisation (``0.0`` → unit scale).
            std: Standard deviation for log-scale initialisation.
            pos_fn: Element-wise function mapping log-scale to positive scale
                factors (default: exp). Alternatives: softplus, relu+ε, etc.
            device: Optional device for parameter initialisation.
            dtype: Optional dtype for parameter initialisation.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._log_scale_mean = float(mean)
        self._log_scale_std = float(std)
        self._pos_fn = pos_fn
        factory = {"device": device, "dtype": dtype}
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory)
        )
        self.log_scale = nn.Parameter(torch.empty(out_features, **factory))
        self.bias = nn.Parameter(torch.empty(out_features, **factory)) if bias else None

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


class SymmetricFactorizedLinear(nn.Linear):
    """Linear layer with symmetry and a sandwich scale factorization.

    Enforces ``W = D @ Sym(A) @ D`` where ``D = diag(exp(s))``, which keeps
    the weight symmetric.  Input and output sizes are identical.
    """

    def __init__(
        self,
        features: int,
        bias: bool = False,
        *,
        mean: float = 0.0,
        std: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the symmetric factorized linear layer.

        Args:
            features: Square matrix dimension.
            bias: Whether to include a bias term.
            mean: Mean for log-scale initialisation (``0.0`` → unit scale).
            std: Standard deviation for log-scale initialisation.
            device: Optional device for weight initialisation.
            dtype: Optional dtype for weight initialisation.
        """
        super().__init__(
            in_features=features,
            out_features=features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        register_symmetric_factorized(self, size=features, mean=mean, std=std)


class SPDFactorizedLinear(nn.Linear):
    """Linear layer with SPD structure and a sandwich scale factorization.

    Enforces ``W = D @ SPD(A) @ D`` where ``D = diag(exp(s))``, which keeps
    the weight symmetric positive-definite.  Input and output sizes are identical.
    """

    def __init__(
        self,
        features: int,
        bias: bool = False,
        *,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the SPD factorized linear layer.

        Args:
            features: Square matrix dimension.
            bias: Whether to include a bias term.
            min_diag: Positive diagonal slack floor for SPD enforcement.
            mean: Mean for log-scale initialisation (``0.0`` → unit scale).
            std: Standard deviation for log-scale initialisation.
            pos_fn: Element-wise positive activation for SPD diagonal enforcement
                (default: softplus).
            device: Optional device for weight initialisation.
            dtype: Optional dtype for weight initialisation.
        """
        super().__init__(
            in_features=features,
            out_features=features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        register_spd_factorized(
            self,
            size=features,
            min_diag=min_diag,
            mean=mean,
            std=std,
            pos_fn=pos_fn,
        )
