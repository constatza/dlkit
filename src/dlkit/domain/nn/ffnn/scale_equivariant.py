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
    EmbeddedFactorizedFFNN,
    _resolve_hidden_size,
)
from dlkit.domain.nn.ffnn.film import FiLMEmbeddedFFNN, FiLMFFNN, VarWidthFiLMFFNN
from dlkit.domain.nn.ffnn.hyper_moe import EmbeddedHyperFFNN, EmbeddedMoEFFNN
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.primitives import (
    DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
    DEFAULT_SCALE_EQUIVARIANT_NORM,
    ConditionedScaleEquivariantWrapper,
    DenseBlockKind,
    DenseLinearKind,
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
        normalize: Literal["batch", "layer"] | None = "layer",
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


# ── Embedded Factorized (FactorizedLinear embedding, body, and regression) ───


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
        skip: bool = True,
        project: bool = True,
        bias: bool = True,
        mean: float = 0.0,
        std: float = 0.1,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedFactorizedFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                skip=skip,
                project=project,
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


# ── Hyper-Connection / Sparse-MoE scale-equivariant variants ────────────────


class ScaleEquivariantEmbeddedHyperFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant Hyper-Connection FFNN with optional projections."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int | None = None,
        num_layers: int,
        num_lanes: int = 2,
        lane_hidden_features: int | None = None,
        project: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            base_model=EmbeddedHyperFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_lanes=num_lanes,
                lane_hidden_features=lane_hidden_features,
                project=project,
                block_kind=block_kind,
                linear_kind=linear_kind,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )


class ScaleEquivariantEmbeddedMoEFFNN(StandardEntryConsumer, ScaleEquivariantWrapper):
    """Scale-equivariant Sparse-MoE FFNN with optional projections.

    Routing diagnostics (``RoutingStats``) are not exposed through this
    wrapper's ``forward`` — the inner ``EmbeddedMoEFFNN`` is always
    constructed with ``return_stats=False`` so that ``ScaleEquivariantWrapper``
    sees a plain output ``Tensor`` to normalize and rescale.
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
        num_experts: int,
        top_k: int = 2,
        expert_hidden_features: int | None = None,
        project: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            base_model=EmbeddedMoEFFNN(
                in_features=in_features,
                out_features=out_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_experts=num_experts,
                top_k=top_k,
                expert_hidden_features=expert_hidden_features,
                project=project,
                block_kind=block_kind,
                linear_kind=linear_kind,
                router_activation=router_activation,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                jitter_noise=jitter_noise,
                return_stats=False,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
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
