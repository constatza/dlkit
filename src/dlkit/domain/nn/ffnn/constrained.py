from __future__ import annotations

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
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
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

    1-1 replica of the pre-2026-05-19 ``ConstantWidthFactorizedFFNN``.
    All ``num_layers`` blocks are ``FactorizedLinear`` wrapped in a
    ``SkipConnection`` with identity skip (valid because in==out throughout).
    No embedding or regression projection — ``in_features`` must equal
    ``out_features``.

    Default activation is GELU to match the original architecture.
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
                "ConstantWidthFactorizedFFNN requires in_features == out_features, "
                f"got {in_features} != {out_features}"
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
    ``ParametricDenseBlock`` instances with no skip connections.

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
                "ConstantWidthSimpleFactorizedFFNN requires in_features == out_features, "
                f"got {in_features} != {out_features}"
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


__all__ = [
    "ConstantWidthFactorizedFFNN",
    "ConstantWidthSimpleFactorizedFFNN",
    "EmbeddedFactorizedFFNN",
    "EmbeddedParametricFFNN",
    "EmbeddedSimpleFactorizedFFNN",
    "EmbeddedSimpleParametricFFNN",
    "FactorizedFFNN",
    "ParametricDenseBlock",
    "SimpleFactorizedFFNN",
    "_resolve_hidden_size",
]
