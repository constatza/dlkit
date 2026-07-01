from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn

from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import (
    StandardEntryConsumer,
)
from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    SkipConnection,
    SoftplusFactorizedLinear,
    build_linear_skip_layer,
)
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import make_norm_layer, resolve_activation


def _resolve_hidden_size(
    hidden_size: int | None,
    in_features: int,
    out_features: int,
) -> int:
    """Return hidden_size, defaulting to in_features when square and omitted."""
    if hidden_size is not None:
        return hidden_size
    if in_features != out_features:
        raise ValueError(
            f"hidden_size must be provided when in_features ({in_features}) "
            f"!= out_features ({out_features})"
        )
    return in_features


class ParametricDenseBlock(nn.Module):
    """Dense block using a caller-supplied linear layer factory.

    Args:
        size: Output dimension (passed to ``layer_factory``).
        in_size: Input dimension for the norm layer. Defaults to ``size`` when
            the block is square; supply when the layer maps ``in_size → size``.
        layer_factory: Callable ``(size) → nn.Module`` that builds the linear layer.
        activation: Element-wise activation applied before the layer.
        normalize: Optional normalisation before activation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability after the layer.
    """

    def __init__(
        self,
        *,
        size: int,
        in_size: int | None = None,
        layer_factory: Callable[[int], nn.Module],
        activation: Callable[[Tensor], Tensor] = nn.functional.relu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        norm_size = in_size if in_size is not None else size
        self.in_features = norm_size
        self.out_features = size
        self.norm = make_norm_layer(normalize, norm_size)
        self.activation = activation
        self.layer = layer_factory(size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.layer(x)
        return self.dropout(x)


class _ConstantWidthParametricBody(nn.Module):
    """Low-level constant-width constrained FFNN body.

    Supports ``num_layers=0`` (empty body that acts as identity).
    """

    def __init__(
        self,
        *,
        size: int,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        _residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.relu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be a positive integer")
        if num_layers < 0:
            raise ValueError("num_layers must be a non-negative integer")

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
            blocks.append(
                SkipConnection(block, build_linear_skip_layer(block)) if _residual else block
            )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _EmbeddedParametricBody(nn.Module):
    """Low-level constrained FFNN with embedding and regression projections."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        _residual: bool = False,
        activation: Callable[[Tensor], Tensor] = nn.functional.relu,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
        regression_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__()
        self.embedding_layer = (
            embedding_factory(in_features, hidden_size)
            if embedding_factory is not None
            else nn.Linear(in_features, hidden_size)
        )
        self.body = _ConstantWidthParametricBody(
            size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=_residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.regression_layer = (
            regression_factory(hidden_size, out_features)
            if regression_factory is not None
            else nn.Linear(hidden_size, out_features)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        x = self.body(x)
        return self.regression_layer(x)


# ── Layer factories ──────────────────────────────────────────────────────────


def _factorized_layer_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int], nn.Module]:
    return lambda n: FactorizedLinear(n, n, bias=bias, mean=mean, std=std)


def _softplus_factorized_layer_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int], nn.Module]:
    return lambda n: SoftplusFactorizedLinear(n, n, bias=bias, mean=mean, std=std)


# softplus(log(e-1)) == 1.0 exactly; user-facing mean=0.0 maps to unit scale at init
_SOFTPLUS_UNIT_MEAN = math.log(math.e - 1)


def _softplus_unit_layer_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int], nn.Module]:
    """Like _softplus_factorized_layer_factory but mean=0.0 → unit scale at init."""
    return lambda n: SoftplusFactorizedLinear(
        n, n, bias=bias, mean=_SOFTPLUS_UNIT_MEAN + mean, std=std
    )


def _factorized_rect_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int, int], nn.Module]:
    """Return a rectangular ``(in_dim, out_dim) -> FactorizedLinear`` factory."""
    return lambda i, o: FactorizedLinear(i, o, bias=bias, mean=mean, std=std)


def _softplus_unit_rect_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int, int], nn.Module]:
    """Return a rectangular ``(in_dim, out_dim) -> SoftplusFactorizedLinear`` factory.

    The mean is shifted by ``log(e-1)`` so that ``softplus(mean) = 1.0`` at
    init when the user-facing ``mean=0.0`` is passed.
    """
    return lambda i, o: SoftplusFactorizedLinear(
        i, o, bias=bias, mean=_SOFTPLUS_UNIT_MEAN + mean, std=std
    )


# ── Public generic builders ──────────────────────────────────────────────────


class EmbeddedParametricFFNN(StandardEntryConsumer, _EmbeddedParametricBody):
    """Residual constrained FFNN with embedding and regression projections."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
        regression_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=True,
            activation=resolve_activation(activation),
            normalize=normalize,
            dropout=dropout,
            embedding_factory=embedding_factory,
            regression_factory=regression_factory,
        )


class EmbeddedSimpleParametricFFNN(StandardEntryConsumer, _EmbeddedParametricBody):
    """Plain constrained FFNN with embedding and regression projections."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        layer_factory: Callable[[int], nn.Module],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
        regression_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=False,
            activation=resolve_activation(activation),
            normalize=normalize,
            dropout=dropout,
            embedding_factory=embedding_factory,
            regression_factory=regression_factory,
        )


# ── Embedded Factorized variants (plain Linear projections) ─────────────────


class EmbeddedFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with factorized body layers."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
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
            ),
            activation=resolve_activation(activation, default="gelu"),
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
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
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
            ),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


# ── Embedded Softplus-Factorized variants (plain Linear projections) ─────────


class EmbeddedSoftplusFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with softplus-factorized body layers.

    The embedding (first) and regression (last) layers are plain ``nn.Linear``
    projections.  Only the constant-width body layers use
    :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`, so that
    ``mean=0.0`` initialises each body layer with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of residual softplus-factorized body blocks.
        bias: Whether body layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
            ),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleSoftplusFactorizedFFNN(EmbeddedSimpleParametricFFNN):
    """Plain (non-residual) embedded FFNN with softplus-factorized body layers.

    The embedding (first) and regression (last) layers are plain ``nn.Linear``
    projections.  Only the constant-width body layers use
    :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`, so that
    ``mean=0.0`` initialises each body layer with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of softplus-factorized body blocks (no skip connections).
        bias: Whether body layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(
                bias=bias,
                mean=mean,
                std=std,
            ),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


# ── FactorizedEnd variants (plain Linear embedding, FactorizedLinear regression) ─


class EmbeddedFactorizedEndFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with factorized body and regression layers.

    The embedding (first) layer is a plain ``nn.Linear`` projection. Both the
    constant-width body and the regression (last) layer use
    :class:`~dlkit.domain.nn.primitives.FactorizedLinear` (exp-based scale).

    Default activation is GELU. Default ``mean=0.0`` → ``exp(0) = 1`` (unit
    scale at init).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of residual factorized body blocks.
        bias: Whether factorized layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` → ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            regression_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleFactorizedEndFFNN(EmbeddedSimpleParametricFFNN):
    """Plain (non-residual) embedded FFNN with factorized body and regression layers.

    The embedding (first) layer is a plain ``nn.Linear`` projection. Both the
    constant-width body (no skip connections) and the regression (last) layer
    use :class:`~dlkit.domain.nn.primitives.FactorizedLinear` (exp-based scale).

    Default activation is GELU. Default ``mean=0.0`` → ``exp(0) = 1`` (unit
    scale at init).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of factorized body blocks (no skip connections).
        bias: Whether factorized layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` → ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            regression_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSoftplusFactorizedEndFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with softplus-factorized body and regression layers.

    The embedding (first) layer is a plain ``nn.Linear`` projection. Both the
    constant-width body and the regression (last) layer use
    :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`, so that
    ``mean=0.0`` initialises each factorized layer with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Default activation is GELU.

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of residual softplus-factorized body blocks.
        bias: Whether factorized layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(bias=bias, mean=mean, std=std),
            regression_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleSoftplusFactorizedEndFFNN(EmbeddedSimpleParametricFFNN):
    """Plain (non-residual) embedded FFNN with softplus-factorized body and regression layers.

    The embedding (first) layer is a plain ``nn.Linear`` projection. Both the
    constant-width body (no skip connections) and the regression (last) layer
    use :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`, so that
    ``mean=0.0`` initialises each factorized layer with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Default activation is GELU.

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of softplus-factorized body blocks (no skip connections).
        bias: Whether factorized layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(bias=bias, mean=mean, std=std),
            regression_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


# ── FullyFactorized variants (FactorizedLinear embedding and regression) ──────


class EmbeddedFullyFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with factorized embedding, body, and regression layers.

    All three layer groups — embedding (first), constant-width body, and
    regression (last) — use :class:`~dlkit.domain.nn.primitives.FactorizedLinear`
    (exp-based scale). No plain ``nn.Linear`` projection is used anywhere.

    Default activation is GELU. Default ``mean=0.0`` → ``exp(0) = 1`` (unit
    scale at init).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of residual factorized body blocks.
        bias: Whether factorized layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` → ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            embedding_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            regression_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleFullyFactorizedFFNN(EmbeddedSimpleParametricFFNN):
    """Plain (non-residual) embedded FFNN with factorized embedding, body, and regression layers.

    All three layer groups — embedding (first), constant-width body (no skip
    connections), and regression (last) — use
    :class:`~dlkit.domain.nn.primitives.FactorizedLinear` (exp-based scale).
    No plain ``nn.Linear`` projection is used anywhere.

    Default activation is GELU. Default ``mean=0.0`` → ``exp(0) = 1`` (unit
    scale at init).

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of factorized body blocks (no skip connections).
        bias: Whether factorized layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` → ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            embedding_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            regression_factory=_factorized_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedFullySoftplusFactorizedFFNN(EmbeddedParametricFFNN):
    """Residual embedded FFNN with softplus-factorized embedding, body, and regression layers.

    All three layer groups — embedding (first), constant-width body, and
    regression (last) — use
    :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`. No plain
    ``nn.Linear`` projection is used anywhere. With ``mean=0.0``, every
    factorized layer initialises with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Default activation is GELU.

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of residual softplus-factorized body blocks.
        bias: Whether factorized layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(bias=bias, mean=mean, std=std),
            embedding_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            regression_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


class EmbeddedSimpleFullySoftplusFactorizedFFNN(EmbeddedSimpleParametricFFNN):
    """Plain (non-residual) embedded FFNN with softplus-factorized embedding, body, and regression.

    All three layer groups — embedding (first), constant-width body (no skip
    connections), and regression (last) — use
    :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear`. No plain
    ``nn.Linear`` projection is used anywhere. With ``mean=0.0``, every
    factorized layer initialises with unit per-neuron scale
    (``softplus(log_scale) ≈ 1``).

    Default activation is GELU.

    Args:
        in_features: Input dimension for the embedding layer.
        out_features: Output dimension of the regression layer.
        hidden_size: Width of all body layers. Required when
            ``in_features != out_features``; defaults to ``in_features``
            when both dimensions are equal.
        num_layers: Number of softplus-factorized body blocks (no skip connections).
        bias: Whether factorized layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation applied before each body layer.
            ``None`` defaults to GELU.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each body layer.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(bias=bias, mean=mean, std=std),
            embedding_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            regression_factory=_softplus_unit_rect_factory(bias=bias, mean=mean, std=std),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
        )


# ── Non-embedded Factorized variants ────────────────────────────────────────


class FactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Residual non-embedded Factorized FFNN.

    First block maps ``in_features → hidden_size`` using a structured Factorized
    layer (no skip — dimensions may differ). Remaining body blocks are square
    ``hidden_size → hidden_size`` with residual connections. Final plain
    ``nn.Linear(hidden_size → out_features)`` regression layer.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.first_block = ParametricDenseBlock(
            size=hidden_size,
            in_size=in_features,
            layer_factory=lambda h: FactorizedLinear(in_features, h, bias=bias, mean=mean, std=std),
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.body = _ConstantWidthParametricBody(
            size=hidden_size,
            num_layers=num_layers - 1,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            _residual=True,
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.regression_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        x = self.body(x)
        return self.regression_layer(x)


class SimpleFactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Plain non-embedded Factorized FFNN.

    First block maps ``in_features → hidden_size`` (no skip). Remaining body
    blocks are square ``hidden_size → hidden_size`` without residual. Final
    plain ``nn.Linear(hidden_size → out_features)`` regression layer.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.first_block = ParametricDenseBlock(
            size=hidden_size,
            in_size=in_features,
            layer_factory=lambda h: FactorizedLinear(in_features, h, bias=bias, mean=mean, std=std),
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.body = _ConstantWidthParametricBody(
            size=hidden_size,
            num_layers=num_layers - 1,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            _residual=False,
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )
        self.regression_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        x = self.body(x)
        return self.regression_layer(x)


# ── Constant-width Factorized variants (pure body, no projection) ───────────


class ConstantWidthFactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Residual constant-width FFNN with factorized body layers.

    All ``num_layers`` blocks are ``FactorizedLinear`` wrapped in a residual
    ``SkipConnection`` with identity skip. No embedding or regression
    projection — every layer including the last uses ``FactorizedLinear``.
    Requires ``in_features == out_features``; for asymmetric inputs use
    :class:`EmbeddedFactorizedFFNN`.

    Default activation is GELU.
    ``log_scale`` initialises as ``N(0.0, 0.1)`` so ``exp(log_scale) ≈ 1``
    at the start of training (unit scale).

    Args:
        in_features: Input and output dimension. Must equal ``out_features``.
        out_features: Output dimension. Must equal ``in_features``.
        num_layers: Number of residual factorized blocks.
        bias: Whether body layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` -> ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each linear layer.
            ``None`` defaults to ``torch.nn.functional.gelu``.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each linear layer.

    Raises:
        ValueError: If ``in_features != out_features``.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"ConstantWidthFactorizedFFNN requires in_features == out_features "
                f"(got {in_features} != {out_features}). "
                "For asymmetric inputs use EmbeddedFactorizedFFNN."
            )
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.body = _ConstantWidthParametricBody(
            size=in_features,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            _residual=True,
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through all residual factorized blocks.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, in_features)``.
        """
        return self.body(x)


class ConstantWidthSimpleFactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Plain (non-residual) constant-width FFNN with factorized body layers.

    Identical to :class:`ConstantWidthFactorizedFFNN` but all blocks are bare
    ``ParametricDenseBlock`` instances with no skip connections. No embedding
    or regression projection — every layer uses ``FactorizedLinear``.
    Requires ``in_features == out_features``; for asymmetric inputs use
    :class:`EmbeddedSimpleFactorizedFFNN`.

    Args:
        in_features: Input and output dimension. Must equal ``out_features``.
        out_features: Output dimension. Must equal ``in_features``.
        num_layers: Number of factorized dense blocks.
        bias: Whether body layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` -> ``exp(0) = 1.0``, unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        activation: Element-wise activation applied before each linear layer.
            ``None`` defaults to ``torch.nn.functional.gelu``.
        normalize: Optional normalisation applied before activation
            (``"batch"`` or ``"layer"``).
        dropout: Dropout probability applied after each linear layer.

    Raises:
        ValueError: If ``in_features != out_features``.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"ConstantWidthSimpleFactorizedFFNN requires in_features == out_features "
                f"(got {in_features} != {out_features}). "
                "For asymmetric inputs use EmbeddedSimpleFactorizedFFNN."
            )
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.body = _ConstantWidthParametricBody(
            size=in_features,
            num_layers=num_layers,
            layer_factory=_factorized_layer_factory(bias=bias, mean=mean, std=std),
            _residual=False,
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through all factorized dense blocks.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, in_features)``.
        """
        return self.body(x)


# ── Constant-width Softplus-Factorized variants ──────────────────────────────


class ConstantWidthSoftplusFactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Residual constant-width FFNN with softplus-factorized layers.

    Uses :class:`~dlkit.domain.nn.primitives.SoftplusFactorizedLinear` so that
    ``mean=0.0`` → ``softplus(log_scale) ≈ 1`` at initialisation (unit scale).
    No embedding or regression projection — every layer uses
    ``SoftplusFactorizedLinear``. Requires ``in_features == out_features``;
    for asymmetric inputs use :class:`EmbeddedSoftplusFactorizedFFNN`.

    Per-block forward: ``GELU(x) → SoftplusFactorizedLinear → x_out + x`` (residual).

    Args:
        in_features: Input and output dimension. Must equal ``out_features``.
        out_features: Output dimension. Must equal ``in_features``.
        num_layers: Number of residual softplus-factorized blocks.
        bias: Whether body layers include a bias term.
        mean: Offset from the softplus unit-scale point
            (``0.0`` → ``log_scale ~ N(log(e-1), std)`` → scale ≈ 1 at init).
        std: Standard deviation for log-scale initialisation.
        activation: Element-wise activation. ``None`` defaults to GELU.
        normalize: Optional normalisation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability.

    Raises:
        ValueError: If ``in_features != out_features``.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        if in_features != out_features:
            raise ValueError(
                f"ConstantWidthSoftplusFactorizedFFNN requires in_features == out_features "
                f"(got {in_features} != {out_features}). "
                "For asymmetric inputs use EmbeddedSoftplusFactorizedFFNN."
            )
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.body = _ConstantWidthParametricBody(
            size=in_features,
            num_layers=num_layers,
            layer_factory=_softplus_unit_layer_factory(bias=bias, mean=mean, std=std),
            _residual=True,
            activation=resolved_activation,
            normalize=normalize,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pass input through all residual softplus-factorized blocks.

        Args:
            x: Input tensor of shape ``(*, in_features)``.

        Returns:
            Output tensor of shape ``(*, in_features)``.
        """
        return self.body(x)


__all__ = [
    "ConstantWidthFactorizedFFNN",
    "ConstantWidthSimpleFactorizedFFNN",
    "ConstantWidthSoftplusFactorizedFFNN",
    "EmbeddedFactorizedEndFFNN",
    "EmbeddedFactorizedFFNN",
    "EmbeddedFullyFactorizedFFNN",
    "EmbeddedFullySoftplusFactorizedFFNN",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleFactorizedEndFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    "EmbeddedSimpleFullyFactorizedFFNN",
    "EmbeddedSimpleFullySoftplusFactorizedFFNN",
    "EmbeddedSimpleParametricFFNN",
    "EmbeddedSimpleSoftplusFactorizedEndFFNN",
    "EmbeddedSimpleSoftplusFactorizedFFNN",
    "EmbeddedSoftplusFactorizedEndFFNN",
    "EmbeddedSoftplusFactorizedFFNN",
    "FactorizedFFNN",
    "ParametricDenseBlock",
    "SimpleFactorizedFFNN",
    "_resolve_hidden_size",
]
