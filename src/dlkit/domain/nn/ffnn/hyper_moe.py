from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from torch import Tensor, nn

from dlkit.domain.nn.contracts import HyperParam, StandardEntryConsumer
from dlkit.domain.nn.contracts import InputSpec as _InputSpec
from dlkit.domain.nn.init import initialize_
from dlkit.domain.nn.primitives import (
    DenseBlockKind,
    DenseLinearKind,
    HyperSequential,
    MoESequential,
    RoutingStats,
    SparseMoE,
    make_dense_block,
    resolve_linear_factory,
)
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation, resolved_activation_name

type DenseBlockFactory = Callable[..., nn.Module]


def _resolve_hidden_size(
    hidden_size: int | None,
    in_features: int,
    out_features: int,
) -> int:
    if hidden_size is not None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        return hidden_size
    if in_features != out_features:
        raise ValueError(
            f"hidden_size must be provided when in_features ({in_features}) "
            f"!= out_features ({out_features})"
        )
    return in_features


def _validate_body_shape(
    *,
    project: bool,
    in_features: int,
    out_features: int,
    hidden_size: int,
) -> None:
    if project:
        return
    if in_features == out_features == hidden_size:
        return
    raise ValueError(
        "project=False requires in_features == out_features == hidden_size "
        f"(got {in_features}, {out_features}, {hidden_size}). "
        "For asymmetric inputs use project=True."
    )


def _make_projection(
    *,
    linear_kind: DenseLinearKind,
    in_features: int,
    out_features: int,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> nn.Module:
    layer = resolve_linear_factory(linear_kind, bias=bias, activation=activation)(
        in_features,
        out_features,
    )
    if linear_kind == "linear":
        initialize_(layer, activation, default="gelu")
    return layer


def _make_body_block(
    *,
    size: int,
    activation: Callable[[Tensor], Tensor],
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
    bias: bool,
    block_kind: DenseBlockKind,
    linear_kind: DenseLinearKind,
    block_factory: DenseBlockFactory | None,
) -> nn.Module:
    if block_factory is not None:
        return block_factory(
            in_features=size,
            out_features=size,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
    return make_dense_block(
        block_kind,
        in_features=size,
        out_features=size,
        activation=activation,
        normalize=normalize,
        dropout=dropout,
        bias=bias,
        linear_kind=linear_kind,
    )


class EmbeddedHyperFFNN(StandardEntryConsumer, nn.Module):
    """Hyper-Connection FFNN with optional input/output projections."""

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
        project: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        _validate_body_shape(
            project=project,
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden,
        )
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")

        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_lanes": num_lanes,
            "project": project,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "block_kind": block_kind if block_factory is None else "custom",
            "linear_kind": linear_kind if block_factory is None else "custom",
        }
        self.embedding_layer = (
            _make_projection(
                linear_kind=linear_kind,
                in_features=in_features,
                out_features=hidden,
                bias=bias,
                activation=activation,
            )
            if project
            else nn.Identity()
        )
        modules = [
            _make_body_block(
                size=hidden,
                activation=resolved_activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
                block_kind=block_kind,
                linear_kind=linear_kind,
                block_factory=block_factory,
            )
            for _ in range(num_layers)
        ]
        self.body = HyperSequential(*modules, num_lanes=num_lanes)
        self.regression_layer = (
            _make_projection(
                linear_kind=linear_kind,
                in_features=hidden,
                out_features=out_features,
                bias=bias,
                activation=activation,
            )
            if project
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.regression_layer(self.body(self.embedding_layer(x)))


class EmbeddedMoEFFNN(StandardEntryConsumer, nn.Module):
    """Sparse-MoE FFNN with optional input/output projections."""

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
        project: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        _validate_body_shape(
            project=project,
            in_features=in_features,
            out_features=out_features,
            hidden_size=hidden,
        )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")

        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "project": project,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "router_activation": router_activation,
            "capacity_factor": capacity_factor,
            "drop_policy": drop_policy,
            "jitter_noise": jitter_noise,
            "return_stats": return_stats,
            "block_kind": block_kind if block_factory is None else "custom",
            "linear_kind": linear_kind if block_factory is None else "custom",
        }
        self.embedding_layer = (
            _make_projection(
                linear_kind=linear_kind,
                in_features=in_features,
                out_features=hidden,
                bias=bias,
                activation=activation,
            )
            if project
            else nn.Identity()
        )
        layers = [
            SparseMoE(
                in_features=hidden,
                experts=[
                    _make_body_block(
                        size=hidden,
                        activation=resolved_activation,
                        normalize=normalize,
                        dropout=dropout,
                        bias=bias,
                        block_kind=block_kind,
                        linear_kind=linear_kind,
                        block_factory=block_factory,
                    )
                    for _ in range(num_experts)
                ],
                top_k=top_k,
                router_activation=router_activation,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                jitter_noise=jitter_noise,
            )
            for _ in range(num_layers)
        ]
        self.body = MoESequential(*layers, return_stats=return_stats)
        self.regression_layer = (
            _make_projection(
                linear_kind=linear_kind,
                in_features=hidden,
                out_features=out_features,
                bias=bias,
                activation=activation,
            )
            if project
            else nn.Identity()
        )

    def forward(
        self,
        x: Tensor,
        *,
        return_stats: bool | None = None,
    ) -> Tensor | tuple[Tensor, list[RoutingStats]]:
        hidden = self.embedding_layer(x)
        result = self.body(hidden, return_stats=return_stats)
        if isinstance(result, tuple):
            body_out, stats = cast(tuple[Tensor, list[RoutingStats]], result)
            return self.regression_layer(body_out), stats
        return self.regression_layer(result)


__all__ = ["EmbeddedHyperFFNN", "EmbeddedMoEFFNN"]
