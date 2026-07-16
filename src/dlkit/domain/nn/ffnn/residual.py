from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from torch import Tensor, nn

from dlkit.domain.nn.contracts import HyperParam, StandardEntryConsumer
from dlkit.domain.nn.contracts import InputSpec as _InputSpec
from dlkit.domain.nn.init import initialize_
from dlkit.domain.nn.primitives import (
    DenseBlockKind,
    DenseLinearKind,
    SkipConnection,
    build_linear_skip_layer,
    make_dense_block,
    residual_branch_scale,
)
from dlkit.domain.nn.types import ActivationName, NormalizerName
from dlkit.domain.nn.utils import resolve_activation, resolved_activation_name

type DenseBlockFactory = Callable[..., nn.Module]


def _make_hidden_block(
    *,
    in_features: int,
    out_features: int,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
    normalize: NormalizerName | None,
    dropout: float,
    bias: bool,
    block_kind: DenseBlockKind,
    linear_kind: DenseLinearKind,
    block_factory: DenseBlockFactory | None,
) -> nn.Module:
    if block_factory is not None:
        return block_factory(
            in_features=in_features,
            out_features=out_features,
            activation=activation,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
    return make_dense_block(
        block_kind,
        in_features=in_features,
        out_features=out_features,
        activation=activation,
        normalize=normalize,
        dropout=dropout,
        bias=bias,
        linear_kind=linear_kind,
    )


class VarWidthFFNN(StandardEntryConsumer, nn.Module):
    """Feed-forward network with explicit per-layer widths.

    Shape diagram:
        ``(B, in_features)``
        -> embedding to ``layers[0]``
        -> hidden transitions across ``layers[1:]``
        -> regression head to ``(B, out_features)``

    More explicitly, for ``layers = [h1, h2, ..., hk]``:
        ``(B, in_features)``
        -> ``(B, h1)``
        -> ``(B, h2)``
        -> ...
        -> ``(B, hk)``
        -> ``(B, out_features)``

    Parameter intuition:
        ``layers`` is the full hidden shape ladder. Changing it changes both
        depth and width profile of the network body.
        ``skip=True`` wraps each hidden transition after the embedding layer in
        a residual skip path; ``skip=False`` keeps the same widths but removes
        the skip connection.
        ``project=False`` treats the class as a body-only stack and requires
        the declared input/output dimensions to match the first/last body widths.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        layers: Hidden-state widths as an explicit list.
        activation: Element-wise activation between layers.
        normalize: Optional normalisation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability between layers.
        bias: Whether linear layers include a bias term.
        skip: If ``True`` (default) each hidden transition is wrapped in a
            ``SkipConnection``; set to ``False`` for a plain dense network.
        project: If ``True`` (default), use embedding and regression projections.
            If ``False``, skip projections and require body-compatible dimensions.
    """

    class InputSpec(_InputSpec):
        pass

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        layers: Sequence[int],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        skip: bool = True,
        project: bool = True,
        block_kind: DenseBlockKind = "dense",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must contain at least one hidden width")
        widths = list(layers)
        if not project and (in_features != widths[0] or out_features != widths[-1]):
            raise ValueError(
                "project=False requires in_features == layers[0] and "
                f"out_features == layers[-1] (got {in_features}, {out_features}, "
                f"{widths[0]}, {widths[-1]})."
            )
        self.num_layers = len(widths) - 1
        self.activation = resolve_activation(activation)
        self.hyperparameters: dict[str, HyperParam] = {
            "num_layers": self.num_layers,
            "activation": resolved_activation_name(activation),
            "normalize": normalize,
            "dropout": dropout,
            "bias": bias,
            "skip": skip,
            "project": project,
            "block_kind": block_kind if block_factory is None else "custom",
            "linear_kind": linear_kind if block_factory is None else "custom",
        }
        branch_scale = residual_branch_scale(self.num_layers)

        self.layers = nn.ModuleList()
        self.embedding_layer = (
            nn.Linear(in_features, widths[0], bias=bias) if project else nn.Identity()
        )
        if project:
            initialize_(self.embedding_layer, activation)

        for i in range(len(widths) - 1):
            block = _make_hidden_block(
                in_features=widths[i],
                out_features=widths[i + 1],
                activation=self.activation,
                normalize=normalize,
                dropout=dropout,
                bias=bias,
                block_kind=block_kind,
                linear_kind=linear_kind,
                block_factory=block_factory,
            )
            layer = (
                SkipConnection(
                    block, build_linear_skip_layer(block, bias=bias), branch_scale=branch_scale
                )
                if skip
                else block
            )
            self.layers.append(layer)

        self.regression_layer = (
            nn.Linear(widths[-1], out_features, bias=bias) if project else nn.Identity()
        )
        if project:
            initialize_(self.regression_layer, activation)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression_layer(x)


class FFNN(VarWidthFFNN):
    """Feed-forward network with constant-width hidden layers.

    Shape diagram:
        ``(B, in_features)``
        -> embedding to ``hidden_size``
        -> ``num_layers`` hidden transitions at width ``hidden_size``
        -> regression head to ``(B, out_features)``

    More explicitly:
        ``(B, in_features)``
        -> ``(B, hidden_size)``
        -> ``(B, hidden_size)`` repeated through the constant-width body
        -> ``(B, out_features)``

    Parameter intuition:
        ``hidden_size`` controls the width of every hidden state.
        ``num_layers`` controls how many hidden ``DenseBlock`` transitions appear
        after the embedding projection.
        ``skip=True`` keeps the same dimensions but adds residual shortcuts
        between hidden transitions.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        hidden_size: Width of all hidden layers. Always required — no default.
        num_layers: Number of hidden ``DenseBlock`` transitions.
        activation: Element-wise activation between layers. Defaults to GELU
            when omitted.
        normalize: Optional normalisation (``"batch"`` or ``"layer"``).
        dropout: Dropout probability between layers.
        bias: Whether linear layers include a bias term.
        skip: If ``True`` (default) each hidden transition is wrapped in a
            ``SkipConnection``; set to ``False`` for a plain dense network.

    Note:
        With ``num_layers=0`` the network is an embedding linear followed
        immediately by the regression linear, with no hidden block between them.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        hidden_size: int,
        num_layers: int,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = "layer",
        dropout: float = 0.0,
        bias: bool = True,
        skip: bool = True,
        project: bool = True,
        block_kind: DenseBlockKind = "dense",
        linear_kind: DenseLinearKind = "linear",
        block_factory: DenseBlockFactory | None = None,
    ) -> None:
        if num_layers < 0:
            raise ValueError("num_layers must be a non-negative integer")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=[hidden_size] * (num_layers + 1),
            activation=resolve_activation(activation, default="gelu"),
            normalize=normalize,
            dropout=dropout,
            bias=bias,
            skip=skip,
            project=project,
            block_kind=block_kind,
            linear_kind=linear_kind,
            block_factory=block_factory,
        )
