from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    SkipConnection,
    SPDFactorizedLinear,
    SPDLinear,
)
from dlkit.domain.nn.utils import make_norm_layer

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeSummary


class ParametricDenseBlock(nn.Module):
    """Dense block using a caller-supplied linear layer factory."""

    def __init__(
        self,
        *,
        size: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = size
        self.out_features = size
        self.norm = make_norm_layer(normalize, size)
        self.activation = activation
        self.layer = layer_factory(size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.layer(x)
        return self.dropout(x)


class _ConstantWidthParametricBody(nn.Module):
    """Low-level constant-width constrained FFNN body."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        _residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be a positive integer")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")

        super().__init__()
        self.residual = _residual

        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block = ParametricDenseBlock(
                size=size,
                layer_factory=layer_factory,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            )
            blocks.append(SkipConnection(block, layer_type="linear") if _residual else block)

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ConstantWidthParametricFFNN(_ConstantWidthParametricBody):
    """Residual constant-width constrained FFNN body."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=True,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSimpleParametricFFNN(_ConstantWidthParametricBody):
    """Plain constant-width constrained FFNN body."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=False,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class _EmbeddedParametricBody(nn.Module):
    """Low-level constrained FFNN with embedding and regression projections."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        _residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_layer = nn.Linear(in_features, hidden_size)
        self.body = _ConstantWidthParametricBody(
            size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=_residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.regression_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        x = self.body(x)
        return self.regression_layer(x)


class EmbeddedParametricFFNN(_EmbeddedParametricBody):
    """Residual constrained FFNN with embedding and regression projections."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=True,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> EmbeddedParametricFFNN:
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )


class EmbeddedSimpleParametricFFNN(_EmbeddedParametricBody):
    """Plain constrained FFNN with embedding and regression projections."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=False,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )

    @classmethod
    def from_shape(cls, shape: ShapeSummary, **kwargs) -> EmbeddedSimpleParametricFFNN:
        return cls(
            in_features=shape.in_features,
            out_features=shape.out_features,
            **kwargs,
        )


def _spd_layer_factory(
    *,
    bias: bool,
    min_diag: float,
    pos_fn: Callable[[Tensor], Tensor],
) -> Callable[[int], nn.Module]:
    return lambda n: SPDLinear(n, bias=bias, min_diag=min_diag, pos_fn=pos_fn)


def _spd_factorized_layer_factory(
    *,
    bias: bool,
    min_diag: float,
    mean: float,
    std: float,
    pos_fn: Callable[[Tensor], Tensor],
) -> Callable[[int], nn.Module]:
    return lambda n: SPDFactorizedLinear(
        n,
        bias=bias,
        min_diag=min_diag,
        mean=mean,
        std=std,
        pos_fn=pos_fn,
    )


def _factorized_layer_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
    pos_fn: Callable[[Tensor], Tensor],
) -> Callable[[int], nn.Module]:
    return lambda n: FactorizedLinear(n, n, bias=bias, mean=mean, std=std, pos_fn=pos_fn)


class ConstantWidthSPDFFNN(ConstantWidthParametricFFNN):
    """Residual constant-width FFNN with SPD-constrained body layers."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
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
            layer_factory=_spd_layer_factory(bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSimpleSPDFFNN(ConstantWidthSimpleParametricFFNN):
    """Plain constant-width FFNN with SPD-constrained body layers."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
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
            layer_factory=_spd_layer_factory(bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSPDFactorizedFFNN(ConstantWidthParametricFFNN):
    """Residual constant-width FFNN with SPD-factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=_spd_factorized_layer_factory(
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSimpleSPDFactorizedFFNN(ConstantWidthSimpleParametricFFNN):
    """Plain constant-width FFNN with SPD-factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            size=size,
            num_layers=num_layers,
            layer_factory=_spd_factorized_layer_factory(
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthFactorizedFFNN(ConstantWidthParametricFFNN):
    """Residual constant-width FFNN with factorized body layers."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
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
            layer_factory=_factorized_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class ConstantWidthSimpleFactorizedFFNN(ConstantWidthSimpleParametricFFNN):
    """Plain constant-width FFNN with factorized body layers."""

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
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
            layer_factory=_factorized_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSPDFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with SPD-constrained body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_spd_layer_factory(bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleSPDFFNN(EmbeddedSimpleParametricFFNN):
    """Plain embedded FFNN with SPD-constrained body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_spd_layer_factory(bias=bias, min_diag=min_diag, pos_fn=pos_fn),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSPDFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with SPD-factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_spd_factorized_layer_factory(
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleSPDFactorizedFFNN(EmbeddedSimpleParametricFFNN):
    """Plain embedded FFNN with SPD-factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_spd_factorized_layer_factory(
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleFactorizedFFNN(EmbeddedSimpleParametricFFNN):
    """Plain embedded FFNN with factorized body layers."""

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
        activation: Callable[[Tensor], Tensor] = nn.functional.gelu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
            ),
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
