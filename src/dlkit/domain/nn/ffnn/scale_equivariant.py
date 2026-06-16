from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Self

import torch.nn.functional as F
from torch import Tensor

from dlkit.domain.nn.contracts import ModelContractSpec, TabulaRSpec
from dlkit.domain.nn.ffnn.constrained import (
    SPDFFNN,
    EmbeddedFactorizedFFNN,
    EmbeddedSimpleFactorizedFFNN,
    EmbeddedSimpleSPDFactorizedFFNN,
    EmbeddedSimpleSPDFFNN,
    EmbeddedSPDFactorizedFFNN,
    EmbeddedSPDFFNN,
    FactorizedFFNN,
    SimpleFactorizedFFNN,
    SimpleSPDFactorizedFFNN,
    SimpleSPDFFNN,
    SPDFactorizedFFNN,
    _resolve_hidden_size,
    _square_contract,
)
from dlkit.domain.nn.ffnn.film import FiLMEmbeddedFFNN, FiLMFFNN, VarWidthFiLMFFNN
from dlkit.domain.nn.ffnn.residual import FFNN
from dlkit.domain.nn.primitives import (
    DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN,
    DEFAULT_SCALE_EQUIVARIANT_NORM,
    DEFAULT_SPD_MIN_DIAG,
    ConditionedScaleEquivariantWrapper,
    ScaleEquivariantWrapper,
)
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import resolve_activation

_DEFAULT_NORM = DEFAULT_SCALE_EQUIVARIANT_NORM
_DEFAULT_EPS_GAIN = DEFAULT_SCALE_EQUIVARIANT_EPS_GAIN


class _ScaleEquivariantBase(ScaleEquivariantWrapper):
    """Wrap a base FFNN with input/output norm scaling to enforce scale equivariance."""


class _ConditionedScaleEquivariantBase(ConditionedScaleEquivariantWrapper):
    """Private base for conditioned scale-equivariant FFNN variants."""


# ── Plain dense (non-structured) ────────────────────────────────────────────


class ScaleEquivariantFFNN(_ScaleEquivariantBase):
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

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


# ── Embedded SPD (all-SPD, square) ──────────────────────────────────────────


class ScaleEquivariantEmbeddedSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded all-SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSPDFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantEmbeddedSimpleSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded all-SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSimpleSPDFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantEmbeddedSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded all-SPDFactorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSPDFactorizedFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantEmbeddedSimpleSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded all-SPDFactorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=EmbeddedSimpleSPDFactorizedFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


# ── Non-embedded SPD (all-SPD, square) ──────────────────────────────────────


class ScaleEquivariantSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual non-embedded all-SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SPDFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantSimpleSPDFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain non-embedded all-SPD FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SimpleSPDFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual non-embedded all-SPDFactorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SPDFactorizedFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


class ScaleEquivariantSimpleSPDFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain non-embedded all-SPDFactorized FFNN."""

    def __init__(
        self,
        *,
        in_features: int,
        num_layers: int,
        bias: bool = False,
        min_diag: float = DEFAULT_SPD_MIN_DIAG,
        mean: float = 0.0,
        std: float = 0.1,
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
        norm: str = _DEFAULT_NORM,
        eps_gain: float = _DEFAULT_EPS_GAIN,
        keep_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            base_model=SimpleSPDFactorizedFFNN(
                in_features=in_features,
                num_layers=num_layers,
                bias=bias,
                min_diag=min_diag,
                mean=mean,
                std=std,
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        return cls(in_features=_square_contract(cls.__name__, contract), **kwargs)


# ── Embedded Factorized (plain Linear projections) ───────────────────────────


class ScaleEquivariantEmbeddedFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual embedded factorized FFNN."""

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
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
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
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


class ScaleEquivariantEmbeddedSimpleFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain embedded factorized FFNN."""

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
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
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
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


# ── Non-embedded Factorized ──────────────────────────────────────────────────


class ScaleEquivariantFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant residual non-embedded factorized FFNN."""

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
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
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
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


class ScaleEquivariantSimpleFactorizedFFNN(_ScaleEquivariantBase):
    """Scale-equivariant plain non-embedded factorized FFNN."""

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
        pos_fn: Callable[[Tensor], Tensor] = F.softplus,
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
                pos_fn=pos_fn,
                activation=resolve_activation(activation),
                normalize=normalize,
                dropout=dropout,
            ),
            norm=norm,
            eps_gain=eps_gain,
            keep_stats=keep_stats,
        )

    @classmethod
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        match contract:
            case TabulaRSpec(in_shape=ins, out_shape=outs):
                return cls(in_features=ins[0], out_features=outs[0], **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires TabulaRSpec, got {type(contract).__name__}"
                )


# ── FiLM-conditioned scale-equivariant variants ──────────────────────────────


class ScaleEquivariantVarWidthFiLMFFNN(_ConditionedScaleEquivariantBase):
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

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from TabulaRSpec.

        Args:
            contract (TabulaRSpec): Shape contract with in_shape and out_shape.
            condition_dim (int): Condition vector dimension.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            Self: Constructed instance.
        """
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )


class ScaleEquivariantFiLMEmbeddedFFNN(_ConditionedScaleEquivariantBase):
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

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from TabulaRSpec.

        Args:
            contract (TabulaRSpec): Shape contract with in_shape and out_shape.
            condition_dim (int): Condition vector dimension.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            Self: Constructed instance.
        """
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )


class ScaleEquivariantFiLMFFNN(_ConditionedScaleEquivariantBase):
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

    @classmethod
    def from_contract(
        cls,
        contract: TabulaRSpec,
        condition_dim: int,
        **kwargs: Any,
    ) -> Self:
        """Construct from TabulaRSpec."""
        return cls(
            in_features=contract.in_shape[0],
            out_features=contract.out_shape[0],
            condition_dim=condition_dim,
            **kwargs,
        )
