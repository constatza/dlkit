from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from torch import Tensor

from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import (
    StandardEntryConsumer,
)
from dlkit.domain.nn.ffnn.constrained import (
    ConstantWidthFactorizedFFNN,
    ConstantWidthSimpleFactorizedFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
    _resolve_hidden_size,
)
from dlkit.domain.nn.ffnn.film import FiLMEmbeddedFFNN, FiLMFFNN, VarWidthFiLMFFNN
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.primitives import (
    DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
    DEFAULT_SCALE_EQUIVARIANT_NORM,
    ConditionedScaleEquivariantWrapper,
    ScaleEquivariantWrapper,
)
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import resolve_activation

_DEFAULT_NORM = DEFAULT_SCALE_EQUIVARIANT_NORM
_DEFAULT_EPS_GAIN = DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN


# ── Plain dense (non-structured) ────────────────────────────────────────────


class ScaleEquivariantFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant residual constant-width FFNN."""

    class InputSpec(_InputSpec):
        pass

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        hidden_size = _resolve_hidden_size(hidden_size, in_features, out_features)
        super().__init__(
            base_model=FFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


# ── Embedded Factorized (plain Linear projections) ───────────────────────────


class ScaleEquivariantEmbeddedFactorizedFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant residual embedded factorized FFNN."""

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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
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
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantEmbeddedSimpleFactorizedFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant plain embedded factorized FFNN."""

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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
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
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


# ── Non-embedded Factorized ──────────────────────────────────────────────────


class ScaleEquivariantFactorizedFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant residual non-embedded factorized FFNN."""

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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=FactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantSimpleFactorizedFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant plain non-embedded factorized FFNN."""

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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SimpleFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


# ── Constant-width Factorized (pure body, no projection) ────────────────────


class ScaleEquivariantConstantWidthFactorizedFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant residual constant-width factorized FFNN.

    Wraps :class:`~dlkit.domain.nn.ffnn.constrained.ConstantWidthFactorizedFFNN`
    with norm-based input/output scaling so that ``f(αx) = α · f(x)`` for any
    scalar ``α > 0``. Requires ``in_features == out_features``.

    Args:
        in_features: Input and output dimension. Must equal ``out_features``.
        out_features: Output dimension. Must equal ``in_features``.
        num_layers: Number of residual factorized blocks.
        bias: Whether body layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` -> unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        norm: Vector norm used for equivariance (``"l2"``, ``"l1"``, ``"linf"``).
        eps_gain: Multiplier on machine epsilon for safe norm division.
        keep_stats: If ``True``, ``forward`` returns ``(output, {"norm": ...})``.
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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantConstantWidthSimpleFactorizedFFNN(
    StandardEntryConsumer, ScaleEquivariantWrapper
):
    """Scale-equivariant plain constant-width factorized FFNN (no skip connections).

    Wraps :class:`~dlkit.domain.nn.ffnn.constrained.ConstantWidthSimpleFactorizedFFNN`
    with norm-based input/output scaling so that ``f(αx) = α · f(x)`` for any
    scalar ``α > 0``. Requires ``in_features == out_features``.

    Args:
        in_features: Input and output dimension. Must equal ``out_features``.
        out_features: Output dimension. Must equal ``in_features``.
        num_layers: Number of factorized dense blocks.
        bias: Whether body layers include a bias term.
        mean: Gaussian mean for ``log_scale`` initialisation
            (``0.0`` -> unit scale at init).
        std: Standard deviation for ``log_scale`` initialisation.
        norm: Vector norm used for equivariance (``"l2"``, ``"l1"``, ``"linf"``).
        eps_gain: Multiplier on machine epsilon for safe norm division.
        keep_stats: If ``True``, ``forward`` returns ``(output, {"norm": ...})``.
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
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=ConstantWidthSimpleFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                num_layers=num_layers,
                bias=bias,
                mean=mean,
                std=std,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


# ── FiLM-conditioned scale-equivariant variants ──────────────────────────────


class ScaleEquivariantVarWidthFiLMFFNN(StandardEntryConsumer, ConditionedScaleEquivariantWrapper):
    """Scale-equivariant variable-width FiLM-conditioned FFNN.

    Scale equivariance applies to the features branch only:
    ``f(αx, c) == α · f(x, c)`` for any scalar α > 0.
    The condition ``c`` is passed through unchanged.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        layers (Sequence[int]): Hidden layer widths.
        norm (str): Vector norm for equivariance (``"l2"``, ``"l1"``, ``"linf"``).
        eps_gain (float): Gain applied to machine epsilon for safe division.
        keep_stats (bool): If True, also return a dict with ``"norm"`` key.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer or None.
        dropout (float): Dropout rate.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        layers: Sequence[int],
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        base = VarWidthFiLMFFNN(
            in_features=in_features,
            out_features=out_features,
            condition_dim=condition_dim,
            layers=layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        super().__init__(base_model=base, norm=norm, eps_gain=eps_gain, keep_stats=keep_stats)


class ScaleEquivariantFiLMEmbeddedFFNN(StandardEntryConsumer, ConditionedScaleEquivariantWrapper):
    """Scale-equivariant FiLM-conditioned embedded constant-width FFNN.

    Scale equivariance applies to the features branch only:
    ``f(αx, c) == α · f(x, c)`` for any scalar α > 0.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        hidden_size (int): Constant hidden width.
        num_layers (int): Number of FiLMResidualBlocks in the body.
        norm (str): Vector norm for equivariance (``"l2"``, ``"l1"``, ``"linf"``).
        eps_gain (float): Gain applied to machine epsilon for safe division.
        keep_stats (bool): If True, also return a dict with ``"norm"`` key.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer or None.
        dropout (float): Dropout rate.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        hidden_size: int,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        base = FiLMEmbeddedFFNN(
            in_features=in_features,
            out_features=out_features,
            condition_dim=condition_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        super().__init__(base_model=base, norm=norm, eps_gain=eps_gain, keep_stats=keep_stats)


class ScaleEquivariantFiLMFFNN(StandardEntryConsumer, ConditionedScaleEquivariantWrapper):
    """Scale-equivariant constant-width FiLM-conditioned FFNN.

    Scale equivariance applies to the features branch only:
    ``f(αx, c) == α · f(x, c)`` for any scalar α > 0.
    The condition ``c`` is passed through unchanged.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        condition_dim (int): Condition vector dimension.
        hidden_size (int): Constant hidden width.
        num_layers (int): Number of hidden FiLM-conditioned transitions.
        norm (str): Vector norm for equivariance (``"l2"``, ``"l1"``, ``"linf"``).
        eps_gain (float): Gain applied to machine epsilon for safe division.
        keep_stats (bool): If True, also return a dict with ``"norm"`` key.
        activation (ActivationName | Callable | None): Activation for DenseBlocks.
        normalize (NormalizerName | None): Norm layer or None.
        dropout (float): Dropout rate.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        condition_dim: int,
        hidden_size: int,
        num_layers: int,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
    ) -> None:
        base = FiLMFFNN(
            in_features=in_features,
            out_features=out_features,
            condition_dim=condition_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
        )
        super().__init__(base_model=base, norm=norm, eps_gain=eps_gain, keep_stats=keep_stats)
