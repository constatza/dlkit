from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.init import initialize_
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import make_norm_layer, resolve_activation

from .factorized_init import resolve_factorized_init
from .parametrized_layers import FactorizedLinear

type DenseBlockKind = Literal["dense", "mlp", "glu", "geglu", "swiglu", "parametric"]
type DenseLinearKind = Literal["linear", "factorized"]
type GateActivationName = Literal["sigmoid", "gelu", "silu", "identity", "none"]


def _identity(x: Tensor) -> Tensor:
    return x


def _resolve_gate_activation(
    name: GateActivationName | Callable[[Tensor], Tensor] | None,
) -> Callable[[Tensor], Tensor]:
    if callable(name):
        return name
    match name:
        case "sigmoid":
            return torch.sigmoid
        case "gelu":
            return F.gelu
        case "silu":
            return F.silu
        case "identity" | "none" | None:
            return _identity
        case _:
            raise ValueError(f"Unsupported gate activation: {name!r}")


def _validate_features(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


class DenseBlock(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initializes a DenseBlock.

        Parameters:
            in_features (int): Number of input features to the layer.
            out_features (int): Number of output features from the layer.
            activation (ActivationName | Callable[[torch.Tensor], torch.Tensor] | None, optional): Activation function or name. Defaults to relu.
            normalize (str | None, optional): Normalization type ('layer', 'batch', or None).
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super().__init__()
        _validate_features("in_features", in_features)
        _validate_features("out_features", out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.norm = make_norm_layer(normalize, in_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        initialize_(self.fc1, activation)
        self.activation = resolve_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return x


class DenseMLPBlock(nn.Module):
    """Two-layer dense feed-forward block for feature-last tensors."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        linear_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        hidden = out_features if hidden_features is None else hidden_features
        _validate_features("in_features", in_features)
        _validate_features("out_features", out_features)
        _validate_features("hidden_features", hidden)
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden
        self.norm = make_norm_layer(normalize, in_features)
        make_linear = (
            (lambda in_f, out_f: nn.Linear(in_f, out_f, bias=bias))
            if linear_factory is None
            else linear_factory
        )
        self.fc1 = make_linear(in_features, hidden)
        self.fc2 = make_linear(hidden, out_features)
        initialize_(self.fc1, activation)
        initialize_(self.fc2, activation)
        self.activation = resolve_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class GatedDenseMLPBlock(nn.Module):
    """GLU-family dense feed-forward block for feature-last tensors."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        gate_activation: GateActivationName | Callable[[Tensor], Tensor] | None = "silu",
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        linear_factory: Callable[[int, int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        hidden = out_features if hidden_features is None else hidden_features
        _validate_features("in_features", in_features)
        _validate_features("out_features", out_features)
        _validate_features("hidden_features", hidden)
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden
        self.norm = make_norm_layer(normalize, in_features)
        make_linear = (
            (lambda in_f, out_f: nn.Linear(in_f, out_f, bias=bias))
            if linear_factory is None
            else linear_factory
        )
        self.value = make_linear(in_features, hidden)
        self.gate = make_linear(in_features, hidden)
        self.proj = make_linear(hidden, out_features)
        initialize_(self.value, activation)
        initialize_(self.gate, gate_activation if isinstance(gate_activation, str) else activation)
        initialize_(self.proj, activation)
        self.gate_activation = _resolve_gate_activation(gate_activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.value(x) * self.gate_activation(self.gate(x))
        x = self.dropout(x)
        x = self.proj(x)
        return self.dropout(x)


class ParametricDenseBlock(nn.Module):
    """Dense block using a caller-supplied linear layer factory."""

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        layer_factory: Callable[[int], nn.Module],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: NormalizerName | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        if hidden_features is not None and hidden_features != out_features:
            raise ValueError("ParametricDenseBlock does not support hidden_features")
        _validate_features("in_features", in_features)
        _validate_features("out_features", out_features)
        if dropout < 0.0:
            raise ValueError("dropout must be non-negative")

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.norm = make_norm_layer(normalize, in_features)
        self.activation = resolve_activation(activation)
        self.layer = layer_factory(out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.activation(x)
        x = self.layer(x)
        return self.dropout(x)


def make_dense_block(
    kind: DenseBlockKind,
    *,
    in_features: int,
    out_features: int,
    hidden_features: int | None = None,
    activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
    normalize: NormalizerName | None = None,
    dropout: float = 0.0,
    bias: bool = True,
    linear_kind: DenseLinearKind = "linear",
    layer_factory: Callable[[int], nn.Module] | None = None,
) -> nn.Module:
    """Build a dense block variant from a string selector."""
    linear_factory = _linear_factory(
        linear_kind,
        bias=bias,
        activation=activation,
    )
    match kind:
        case "dense":
            if linear_kind == "factorized":
                return ParametricDenseBlock(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=hidden_features,
                    layer_factory=lambda n: linear_factory(in_features, n),
                    activation=activation,
                    normalize=normalize,
                    dropout=dropout,
                    bias=bias,
                )
            return DenseBlock(
                in_features=in_features,
                out_features=out_features,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
            )
        case "mlp":
            return DenseMLPBlock(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
                linear_factory=linear_factory,
            )
        case "glu" | "geglu" | "swiglu":
            gate_activation: GateActivationName
            match kind:
                case "glu":
                    gate_activation = "sigmoid"
                case "geglu":
                    gate_activation = "gelu"
                case "swiglu":
                    gate_activation = "silu"
            return GatedDenseMLPBlock(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                activation=activation,
                gate_activation=gate_activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
                linear_factory=linear_factory,
            )
        case "parametric":
            if layer_factory is None:
                raise ValueError("layer_factory is required for kind='parametric'")
            return ParametricDenseBlock(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_features,
                layer_factory=layer_factory,
                activation=activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
            )
        case _:
            supported = ("dense", "mlp", "glu", "geglu", "swiglu", "parametric")
            raise ValueError(f"Unsupported dense block kind {kind!r}; supported: {supported}")


def _linear_factory(
    kind: DenseLinearKind,
    *,
    bias: bool,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
) -> Callable[[int, int], nn.Module]:
    match kind:
        case "linear":
            return lambda in_features, out_features: nn.Linear(
                in_features,
                out_features,
                bias=bias,
            )
        case "factorized":
            init = resolve_factorized_init(activation)
            return lambda in_features, out_features: FactorizedLinear(
                in_features,
                out_features,
                bias=bias,
                mean=init.log_scale_mean,
                std=init.log_scale_std,
                kaiming_a=init.kaiming_a,
            )
