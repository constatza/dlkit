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
from dlkit.domain.nn.init import initialize_
from dlkit.domain.nn.primitives import (
    FactorizedLinear,
    ParametricDenseBlock,
    SkipConnection,
    build_linear_skip_layer,
)
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation


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
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be a positive integer")
        if num_layers < 0:
            raise ValueError("num_layers must be a non-negative integer")

        super().__init__()
        self.residual = _residual
        # GPT-2 appendix (Radford et al. 2019): scale each residual branch by
        # 1/sqrt(2*num_layers) to counteract geometric variance growth across depth.
        branch_scale = 1.0 / math.sqrt(2 * num_layers) if num_layers > 0 else 1.0

        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            block = ParametricDenseBlock(
                in_features=size,
                out_features=size,
                layer_factory=layer_factory,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            )
            blocks.append(
                SkipConnection(block, build_linear_skip_layer(block), branch_scale=branch_scale)
                if _residual
                else block
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
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        embedding_factory: Callable[[int, int], nn.Module] | None = None,
        regression_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__()
        if embedding_factory is not None:
            self.embedding_layer = embedding_factory(in_features, hidden_size)
        else:
            self.embedding_layer = nn.Linear(in_features, hidden_size)
            initialize_(self.embedding_layer, activation)
        self.body = _ConstantWidthParametricBody(
            size=hidden_size,
            num_layers=num_layers,
            layer_factory=layer_factory,
            _residual=_residual,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        if regression_factory is not None:
            self.regression_layer = regression_factory(hidden_size, out_features)
        else:
            self.regression_layer = nn.Linear(hidden_size, out_features)
            initialize_(self.regression_layer, activation)

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


def _factorized_rect_factory(
    *,
    bias: bool,
    mean: float,
    std: float,
) -> Callable[[int, int], nn.Module]:
    """Return a rectangular ``(in_dim, out_dim) -> FactorizedLinear`` factory."""
    return lambda i, o: FactorizedLinear(i, o, bias=bias, mean=mean, std=std)


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
        normalize: Literal["batch", "layer"] | None = "layer",
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
        normalize: Literal["batch", "layer"] | None = "layer",
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


# ── Embedded factorized variants (FactorizedLinear embedding and regression) ──


class EmbeddedFactorizedFFNN(EmbeddedParametricFFNN):
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
        normalize: Literal["batch", "layer"] | None = "layer",
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


class EmbeddedSimpleFactorizedFFNN(EmbeddedSimpleParametricFFNN):
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
        normalize: Literal["batch", "layer"] | None = "layer",
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


# ── Non-embedded Factorized variants ────────────────────────────────────────


class FactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Residual non-embedded Factorized FFNN.

    First block maps ``in_features → hidden_size`` using a structured Factorized
    layer (no skip — dimensions may differ). Remaining body blocks are square
    ``hidden_size → hidden_size`` with residual connections. Final
    ``FactorizedLinear(hidden_size → out_features)`` regression layer.
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
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.first_block = ParametricDenseBlock(
            in_features=in_features,
            out_features=hidden_size,
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
        self.regression_layer = _factorized_rect_factory(bias=bias, mean=mean, std=std)(
            hidden_size, out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.first_block(x)
        x = self.body(x)
        return self.regression_layer(x)


class SimpleFactorizedFFNN(StandardEntryConsumer, nn.Module):
    """Plain non-embedded Factorized FFNN.

    First block maps ``in_features → hidden_size`` (no skip). Remaining body
    blocks are square ``hidden_size → hidden_size`` without residual. Final
    ``FactorizedLinear(hidden_size → out_features)`` regression layer.
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
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        resolved_activation = resolve_activation(activation, default="gelu")
        super().__init__()
        self.first_block = ParametricDenseBlock(
            in_features=in_features,
            out_features=hidden_size,
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
        self.regression_layer = _factorized_rect_factory(bias=bias, mean=mean, std=std)(
            hidden_size, out_features
        )

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
        normalize: Literal["batch", "layer"] | None = "layer",
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
        normalize: Literal["batch", "layer"] | None = "layer",
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
