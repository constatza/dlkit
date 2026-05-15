from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

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
    _resolve_hidden_size,
)
from dlkit.domain.nn.ffnn.residual import ConstantWidthFFNN, FeedForwardNN
from dlkit.domain.nn.ffnn.simple import ConstantWidthSimpleFFNN, SimpleFeedForwardNN
from dlkit.domain.nn.primitives import (
    DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
    DEFAULT_SCALE_EQUIVARIANT_NORM,
    ScaleEquivariantWrapper,
    shape_aware_kwargs,
)

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary

_DEFAULT_NORM = DEFAULT_SCALE_EQUIVARIANT_NORM
_DEFAULT_EPS_GAIN = DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN


class _ScaleEquivariantBase(ScaleEquivariantWrapper):
    """Wrap a base FFNN with input/output norm scaling to enforce scale equivariance."""


class _SquareScaleEquivariantBase(_ScaleEquivariantBase):
    """Scale-equivariant wrapper for architecturally-square constrained networks.

    Implements ``ShapeConsumer`` so the build system can inject ``in_features``
    from dataset shape. Output dimension equals input dimension by construction.
    """

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs: Any) -> Self:
        kwargs["in_features"] = shape.in_features
        return cls(**kwargs)


def _default_activation(
    activation: Callable[[Tensor], Tensor] | None,
) -> Callable[[Tensor], Tensor]:
    return activation if activation is not None else nn.functional.gelu


class ScaleEquivariantConstantWidthFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual constant-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
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
        return cls(**shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantConstantWidthSimpleFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain constant-width FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


class ScaleEquivariantConstantWidthSPDFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant residual constant-width SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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


class ScaleEquivariantConstantWidthSimpleSPDFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant plain constant-width SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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


class ScaleEquivariantConstantWidthSPDFactorizedFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant residual constant-width SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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


class ScaleEquivariantConstantWidthSimpleSPDFactorizedFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant plain constant-width SPD-factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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


class ScaleEquivariantConstantWidthFactorizedFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant residual constant-width factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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


class ScaleEquivariantConstantWidthSimpleFactorizedFFNN(_SquareScaleEquivariantBase):
    """Scale-equivariant plain constant-width factorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
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
        size = in_features
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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))


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
        return cls(**shape_aware_kwargs(shape, kwargs))
