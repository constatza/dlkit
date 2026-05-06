from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.ffnn.constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    ConstantWidthSimpleSPDFactorizedFFNN,
    ConstantWidthSimpleSPDFFNN,
    ConstantWidthSPDFactorizedFFNN,
    ConstantWidthSPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
)
from dlkit.domain.nn.ffnn.residual import ConstantWidthFFNN, FeedForwardNN
from dlkit.domain.nn.ffnn.simple import ConstantWidthSimpleFFNN, SimpleFeedForwardNN

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary

_DEFAULT_NORM = "l2"
_DEFAULT_EPS_GAIN = 10.0


class _ScaleEquivariantBase(nn.Module):
    """Wrap a base FFNN with input/output norm scaling to enforce scale equivariance."""

    SUPPORTED_NORMS = {"l2", "l1", "linf"}

    def __init__(
        self,
        *,
        base_model: nn.Module,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
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
        finfo = torch.finfo(x.dtype)
        return float(gain * finfo.eps)

    def _vector_norm(self, x: Tensor) -> Tensor:
        match self.norm:
            case "l2":
                return torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
            case "l1":
                return torch.linalg.vector_norm(x, ord=1, dim=-1, keepdim=True)
            case _:
                return torch.linalg.vector_norm(x, ord=float("inf"), dim=-1, keepdim=True)

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
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


def _default_activation(
    activation: Callable[[Tensor], Tensor] | None,
) -> Callable[[Tensor], Tensor]:
    return activation if activation is not None else nn.functional.gelu


def _shape_aware_kwargs(shape: ShapeSummary, kwargs: dict[str, Any]) -> dict[str, Any]:
    shape_kwargs = dict(kwargs)
    shape_kwargs.pop("in_features", None)
    shape_kwargs.pop("out_features", None)
    return {
        "in_features": shape.in_features,
        "out_features": shape.out_features,
        **shape_kwargs,
    }


class ScaleEquivariantConstantWidthFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual constant-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantConstantWidthFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantConstantWidthSimpleFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain constant-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSimpleFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantConstantWidthSimpleFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantFeedForwardNN(_ScaleEquivariantBase):
    """Scale-equivariant residual variable-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=FeedForwardNN(
                in_features=in_features,
                out_features=out_features,
                layers=layers,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantFeedForwardNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantSimpleFeedForwardNN(_ScaleEquivariantBase):
    """Scale-equivariant plain variable-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SimpleFeedForwardNN(
                in_features=in_features,
                out_features=out_features,
                layers=layers,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantSimpleFeedForwardNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantConstantWidthSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual constant-width SPD FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSPDFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthSimpleSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain constant-width SPD FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSimpleSPDFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual constant-width SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSPDFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain constant-width SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSimpleSPDFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual constant-width factorized FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthSimpleFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain constant-width factorized FFNN."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSimpleFactorizedFFNN(
                size=size,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantEmbeddedSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
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
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantEmbeddedSPDFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantEmbeddedSimpleSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSimpleSPDFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantEmbeddedSimpleSPDFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantEmbeddedSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
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
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantEmbeddedSPDFactorizedFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = 1e-4,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSimpleSPDFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(
        cls, shape: ShapeSummary, **kwargs
    ) -> ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantEmbeddedFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
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
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> ScaleEquivariantEmbeddedFactorizedFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantEmbeddedSimpleFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = torch.exp,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSimpleFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=_default_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_shape(
        cls, shape: ShapeSummary, **kwargs
    ) -> ScaleEquivariantEmbeddedSimpleFactorizedFFNN:
        return cls(**_shape_aware_kwargs(shape, kwargs))
