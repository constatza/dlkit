from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn

from dlkit.domain.nn.contracts import HyperParam, StandardEntryConsumer
from dlkit.domain.nn.contracts import InputSpec as _InputSpec
from dlkit.domain.nn.init import initialize_
from dlkit.domain.nn.primitives import (
    DenseBlockKind,
    DenseLinearKind,
    FactorizedLinear,
    HyperSequential,
    MoESequential,
    ParametricDenseBlock,
    RoutingStats,
    SparseMoE,
    make_dense_block,
    resolve_factorized_init,
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


def _validate_square(in_features: int, out_features: int) -> None:
    if in_features != out_features:
        raise ValueError(
            "constant-width architectures require "
            f"in_features == out_features, got {in_features} != {out_features}"
        )


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=bias)
    initialize_(layer, activation, default="gelu")
    return layer


def _linear_layer_factory(
    *,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> Callable[[int], nn.Module]:
    return lambda n: _linear(n, n, bias=bias, activation=activation)


def _factorized_linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> FactorizedLinear:
    init = resolve_factorized_init(activation, default="gelu")
    return FactorizedLinear(
        in_features,
        out_features,
        bias=bias,
        mean=init.log_scale_mean,
        std=init.log_scale_std,
        kaiming_a=init.kaiming_a,
    )


def _factorized_layer_factory(
    *,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> Callable[[int], nn.Module]:
    return lambda n: _factorized_linear(n, n, bias=bias, activation=activation)


def _linear_block(
    *,
    size: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
) -> ParametricDenseBlock:
    return ParametricDenseBlock(
        in_features=size,
        out_features=size,
        layer_factory=_linear_layer_factory(bias=bias, activation=activation_for_init),
        activation=activation,
        normalize=normalize,
        dropout=dropout,
    )


def _make_selected_block(
    *,
    size: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
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
    if block_kind == "parametric":
        make_block = _factorized_block if linear_kind == "factorized" else _linear_block
        return make_block(
            size=size,
            bias=bias,
            activation=activation,
            activation_for_init=activation_for_init,
            normalize=normalize,
            dropout=dropout,
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


def _factorized_block(
    *,
    size: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
) -> ParametricDenseBlock:
    return ParametricDenseBlock(
        in_features=size,
        out_features=size,
        layer_factory=_factorized_layer_factory(bias=bias, activation=activation_for_init),
        activation=activation,
        normalize=normalize,
        dropout=dropout,
    )


def _make_linear_experts(
    *,
    size: int,
    num_experts: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
    block_kind: DenseBlockKind = "parametric",
    linear_kind: DenseLinearKind = "linear",
    block_factory: DenseBlockFactory | None = None,
) -> list[nn.Module]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return [
        _make_selected_block(
            size=size,
            bias=bias,
            activation=activation,
            activation_for_init=activation_for_init,
            normalize=normalize,
            dropout=dropout,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
        )
        for _ in range(num_experts)
    ]


def _make_factorized_experts(
    *,
    size: int,
    num_experts: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
) -> list[nn.Module]:
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    return [
        _factorized_block(
            size=size,
            bias=bias,
            activation=activation,
            activation_for_init=activation_for_init,
            normalize=normalize,
            dropout=dropout,
        )
        for _ in range(num_experts)
    ]


def _moe_hyperparameters(
    *,
    num_layers: int,
    num_experts: int,
    top_k: int,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
    bias: bool,
    router_activation: Literal["softmax", "normalized_sigmoid"],
    capacity_factor: float | None,
    drop_policy: Literal["none", "drop"],
    jitter_noise: float,
    return_stats: bool,
    hidden_size: int | None = None,
    block_kind: DenseBlockKind = "parametric",
    linear_kind: DenseLinearKind = "linear",
    block_factory: DenseBlockFactory | None = None,
) -> dict[str, HyperParam]:
    params: dict[str, HyperParam] = {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
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
    if hidden_size is not None:
        params["hidden_size"] = hidden_size
    return params


def _make_moe_layers(
    *,
    size: int,
    num_layers: int,
    experts_factory: Callable[[], list[nn.Module]],
    top_k: int,
    router_activation: Literal["softmax", "normalized_sigmoid"],
    capacity_factor: float | None,
    drop_policy: Literal["none", "drop"],
    jitter_noise: float,
) -> list[SparseMoE]:
    return [
        SparseMoE(
            in_features=size,
            experts=experts_factory(),
            top_k=top_k,
            router_activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
        )
        for _ in range(num_layers)
    ]


def _make_linear_moe_layers(
    *,
    size: int,
    num_layers: int,
    num_experts: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
    block_kind: DenseBlockKind,
    linear_kind: DenseLinearKind,
    block_factory: DenseBlockFactory | None,
    top_k: int,
    router_activation: Literal["softmax", "normalized_sigmoid"],
    capacity_factor: float | None,
    drop_policy: Literal["none", "drop"],
    jitter_noise: float,
) -> list[SparseMoE]:
    return _make_moe_layers(
        size=size,
        num_layers=num_layers,
        experts_factory=lambda: _make_linear_experts(
            size=size,
            num_experts=num_experts,
            bias=bias,
            activation=activation,
            activation_for_init=activation_for_init,
            normalize=normalize,
            dropout=dropout,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
        ),
        top_k=top_k,
        router_activation=router_activation,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
        jitter_noise=jitter_noise,
    )


def _make_factorized_moe_layers(
    *,
    size: int,
    num_layers: int,
    num_experts: int,
    bias: bool,
    activation: Callable[[Tensor], Tensor],
    activation_for_init: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: Literal["batch", "layer"] | None,
    dropout: float,
    top_k: int,
    router_activation: Literal["softmax", "normalized_sigmoid"],
    capacity_factor: float | None,
    drop_policy: Literal["none", "drop"],
    jitter_noise: float,
) -> list[SparseMoE]:
    return _make_moe_layers(
        size=size,
        num_layers=num_layers,
        experts_factory=lambda: _make_factorized_experts(
            size=size,
            num_experts=num_experts,
            bias=bias,
            activation=activation,
            activation_for_init=activation_for_init,
            normalize=normalize,
            dropout=dropout,
        ),
        top_k=top_k,
        router_activation=router_activation,
        capacity_factor=capacity_factor,
        drop_policy=drop_policy,
        jitter_noise=jitter_noise,
    )


class ConstantWidthHyper(StandardEntryConsumer, nn.Module):
    """Square Hyper-Connection FFNN with ordinary linear hidden transforms."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        num_lanes: int = 2,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        _validate_square(in_features, out_features)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "num_layers": num_layers,
            "num_lanes": num_lanes,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "block_kind": block_kind if block_factory is None else "custom",
            "linear_kind": linear_kind if block_factory is None else "custom",
        }
        modules = [
            _make_selected_block(
                size=in_features,
                bias=bias,
                activation=resolved_activation,
                activation_for_init=activation,
                normalize=normalize,
                dropout=dropout,
                block_kind=block_kind,
                linear_kind=linear_kind,
                block_factory=block_factory,
            )
            for _ in range(num_layers)
        ]
        self.body = HyperSequential(*modules, num_lanes=num_lanes)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class EmbeddedHyper(StandardEntryConsumer, nn.Module):
    """Rectangular-safe Hyper-Connection FFNN with ordinary linear projections."""

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_lanes": num_lanes,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "block_kind": block_kind if block_factory is None else "custom",
            "linear_kind": linear_kind if block_factory is None else "custom",
        }
        self.embedding_layer = _linear(in_features, hidden, bias=bias, activation=activation)
        modules = [
            _make_selected_block(
                size=hidden,
                bias=bias,
                activation=resolved_activation,
                activation_for_init=activation,
                normalize=normalize,
                dropout=dropout,
                block_kind=block_kind,
                linear_kind=linear_kind,
                block_factory=block_factory,
            )
            for _ in range(num_layers)
        ]
        self.body = HyperSequential(*modules, num_lanes=num_lanes)
        self.regression_layer = _linear(hidden, out_features, bias=bias, activation=activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.regression_layer(self.body(self.embedding_layer(x)))


class ConstantWidthHyperFactorized(StandardEntryConsumer, nn.Module):
    """Square Hyper-Connection FFNN with factorized hidden transforms only."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        num_lanes: int = 2,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        _validate_square(in_features, out_features)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "num_layers": num_layers,
            "num_lanes": num_lanes,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
        }
        modules = [
            _factorized_block(
                size=in_features,
                bias=bias,
                activation=resolved_activation,
                activation_for_init=activation,
                normalize=normalize,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.body = HyperSequential(*modules, num_lanes=num_lanes)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class EmbeddedHyperFactorized(StandardEntryConsumer, nn.Module):
    """Rectangular-safe Hyper-Connection FFNN with all-factorized projections."""

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_lanes": num_lanes,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
        }
        self.embedding_layer = _factorized_linear(
            in_features, hidden, bias=bias, activation=activation
        )
        modules = [
            _factorized_block(
                size=hidden,
                bias=bias,
                activation=resolved_activation,
                activation_for_init=activation,
                normalize=normalize,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        self.body = HyperSequential(*modules, num_lanes=num_lanes)
        self.regression_layer = _factorized_linear(
            hidden, out_features, bias=bias, activation=activation
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.regression_layer(self.body(self.embedding_layer(x)))


class ConstantWidthMoEFactorized(StandardEntryConsumer, nn.Module):
    """Square sparse-MoE FFNN whose experts are constant-width factorized blocks."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        num_experts: int,
        top_k: int = 2,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
    ) -> None:
        super().__init__()
        _validate_square(in_features, out_features)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "router_activation": router_activation,
            "capacity_factor": capacity_factor,
            "drop_policy": drop_policy,
            "jitter_noise": jitter_noise,
            "return_stats": return_stats,
        }
        layers = [
            SparseMoE(
                in_features=in_features,
                experts=_make_factorized_experts(
                    size=in_features,
                    num_experts=num_experts,
                    bias=bias,
                    activation=resolved_activation,
                    activation_for_init=activation,
                    normalize=normalize,
                    dropout=dropout,
                ),
                top_k=top_k,
                router_activation=router_activation,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                jitter_noise=jitter_noise,
            )
            for _ in range(num_layers)
        ]
        self.body = MoESequential(*layers, return_stats=return_stats)

    def forward(
        self,
        x: Tensor,
        *,
        return_stats: bool | None = None,
    ) -> Tensor | tuple[Tensor, list[RoutingStats]]:
        return self.body(x, return_stats=return_stats)


class ConstantWidthMoE(StandardEntryConsumer, nn.Module):
    """Square sparse-MoE FFNN whose experts use ordinary linear kernels."""

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_layers: int,
        num_experts: int,
        top_k: int = 2,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        _validate_square(in_features, out_features)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters = _moe_hyperparameters(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
            router_activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
            return_stats=return_stats,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
        )
        layers = _make_linear_moe_layers(
            size=in_features,
            num_layers=num_layers,
            num_experts=num_experts,
            bias=bias,
            activation=resolved_activation,
            activation_for_init=activation,
            normalize=normalize,
            dropout=dropout,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
            top_k=top_k,
            router_activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
        )
        self.body = MoESequential(*layers, return_stats=return_stats)

    def forward(
        self,
        x: Tensor,
        *,
        return_stats: bool | None = None,
    ) -> Tensor | tuple[Tensor, list[RoutingStats]]:
        return self.body(x, return_stats=return_stats)


class EmbeddedMoEFactorized(StandardEntryConsumer, nn.Module):
    """Rectangular-safe sparse-MoE FFNN with all-factorized projections."""

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters: dict[str, HyperParam] = {
            "hidden_size": hidden,
            "num_layers": num_layers,
            "num_experts": num_experts,
            "top_k": top_k,
            "activation": resolved_activation_name(activation, default="gelu"),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "router_activation": router_activation,
            "capacity_factor": capacity_factor,
            "drop_policy": drop_policy,
            "jitter_noise": jitter_noise,
            "return_stats": return_stats,
        }
        self.embedding_layer = _factorized_linear(
            in_features, hidden, bias=bias, activation=activation
        )
        layers = [
            SparseMoE(
                in_features=hidden,
                experts=_make_factorized_experts(
                    size=hidden,
                    num_experts=num_experts,
                    bias=bias,
                    activation=resolved_activation,
                    activation_for_init=activation,
                    normalize=normalize,
                    dropout=dropout,
                ),
                top_k=top_k,
                router_activation=router_activation,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                jitter_noise=jitter_noise,
            )
            for _ in range(num_layers)
        ]
        self.body = MoESequential(*layers, return_stats=return_stats)
        self.regression_layer = _factorized_linear(
            hidden, out_features, bias=bias, activation=activation
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
            body_out, stats = result
            return self.regression_layer(body_out), stats
        return self.regression_layer(result)


class EmbeddedMoE(StandardEntryConsumer, nn.Module):
    """Rectangular-safe sparse-MoE FFNN with ordinary linear projections."""

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
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        router_activation: Literal["softmax", "normalized_sigmoid"] = "softmax",
        capacity_factor: float | None = None,
        drop_policy: Literal["none", "drop"] = "none",
        jitter_noise: float = 0.0,
        return_stats: bool = False,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        hidden = _resolve_hidden_size(hidden_size, in_features, out_features)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        resolved_activation = resolve_activation(activation, default="gelu")
        self.num_layers = num_layers
        self.activation = resolved_activation
        self.hyperparameters = _moe_hyperparameters(
            hidden_size=hidden,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
            router_activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
            return_stats=return_stats,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
        )
        self.embedding_layer = _linear(in_features, hidden, bias=bias, activation=activation)
        layers = _make_linear_moe_layers(
            size=hidden,
            num_layers=num_layers,
            num_experts=num_experts,
            bias=bias,
            activation=resolved_activation,
            activation_for_init=activation,
            normalize=normalize,
            dropout=dropout,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
            top_k=top_k,
            router_activation=router_activation,
            capacity_factor=capacity_factor,
            drop_policy=drop_policy,
            jitter_noise=jitter_noise,
        )
        self.body = MoESequential(*layers, return_stats=return_stats)
        self.regression_layer = _linear(hidden, out_features, bias=bias, activation=activation)

    def forward(
        self,
        x: Tensor,
        *,
        return_stats: bool | None = None,
    ) -> Tensor | tuple[Tensor, list[RoutingStats]]:
        hidden = self.embedding_layer(x)
        result = self.body(hidden, return_stats=return_stats)
        if isinstance(result, tuple):
            body_out, stats = result
            return self.regression_layer(body_out), stats
        return self.regression_layer(result)


__all__ = [
    "ConstantWidthHyper",
    "ConstantWidthHyperFactorized",
    "ConstantWidthMoE",
    "ConstantWidthMoEFactorized",
    "EmbeddedHyper",
    "EmbeddedHyperFactorized",
    "EmbeddedMoE",
    "EmbeddedMoEFactorized",
]
