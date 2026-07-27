"""GNN composites combining Hyper-Connection and Sparse-MoE primitives with graph convs.

The GNN analog of ``ffnn/hyper_moe.py``'s ``EmbeddedHyperFFNN``/``EmbeddedMoEFFNN``:
fixed input/output projections (via :class:`~dlkit.domain.nn.graph.projection_networks.GProjection`)
wrap a message-passing body that interleaves graph convolutions with either
Hyper-Connection lane mixing or Sparse-MoE routing.

Both composites deliberately fix the residual mechanism of their inner conv
sublayer rather than exposing it as a caller-facing toggle -- see each
class's docstring for why.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import Tensor, nn

from dlkit.domain.nn.primitives import (
    DenseBlockKind,
    DenseLinearKind,
    GraphHyperSequential,
    SparseMoE,
    make_dense_block,
    residual_branch_scale,
)
from dlkit.domain.nn.primitives.moe import DropPolicy, RouterActivation
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation

from .conv import GraphConvKind, resolve_graph_conv_factory
from .projection_networks import GProjection

__all__ = ["EmbeddedHyperGraphNetwork", "EmbeddedMoEGraphNetwork"]


class _ActivatedGraphConv(nn.Module):
    """Applies a post-conv activation, mirroring every other message-passing body.

    ``GraphHyperSequential``/``GraphHyperConnection`` call each wrapped module
    once per lane with no activation applied in between -- unlike
    ``_GraphMessageBase.forward`` in :mod:`dlkit.domain.nn.graph.message`,
    which applies the activation itself in its per-layer loop. Wrapping each
    raw conv adapter in this module before handing it to
    :class:`~dlkit.domain.nn.primitives.stacks.GraphHyperSequential` restores
    that per-layer nonlinearity, keeping semantic parity with the rest of the
    graph package.

    Args:
        conv: A raw graph-conv adapter implementing
            ``forward(x, edge_index, edge_attr=None) -> Tensor``.
        activation: Activation callable applied to the conv's output.
    """

    def __init__(self, *, conv: nn.Module, activation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.conv = conv
        self.activation = activation

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Apply the wrapped conv followed by the activation.

        Args:
            x: Node feature tensor of shape ``(num_nodes, hidden_size)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Optional edge features; validity is kind-specific.

        Returns:
            Tensor: Activated node embeddings of shape
                ``(num_nodes, hidden_size)``.
        """
        return self.activation(self.conv(x, edge_index, edge_attr))


def _build_hyper_message_module(
    *,
    conv_kind: GraphConvKind,
    hidden_size: int,
    num_layers: int,
    num_lanes: int,
    edge_dim: int | None,
    activation: ActivationName | Callable[[Tensor], Tensor] | None,
    dropout: float,
    **conv_kwargs,
) -> GraphHyperSequential:
    """Build the default Hyper-Connection message-passing module.

    The per-layer conv adapters are built with ``residual=False`` (plain),
    never ``residual=True``. ``GraphHyperConnection``'s lane-mixing matrices
    are identity-biased at init specifically so a plain, non-residual wrapped
    module starts as an exact multi-lane residual identity (the same
    principle as ``HyperConnection`` wrapping plain FFNN dense blocks in
    ``EmbeddedHyperFFNN``). If the conv adapter also had its own residual
    (GATv2's native one, or GCN/SAGE's adapter-level ``x + conv(...)``), the
    stack would get residual-on-residual, breaking the identity-at-init
    property the Hyper-Connection design relies on for stable deep stacking.

    Args:
        conv_kind: Graph convolution strategy used for every layer.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of stacked graph-conv layers (lanes).
        num_lanes: Number of Hyper-Connection residual lanes.
        edge_dim: Optional edge feature dimensionality.
        activation: Activation function applied after each conv layer.
        dropout: Dropout probability; semantics are kind-specific.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to every layer.

    Returns:
        GraphHyperSequential: The built Hyper-Connection message-passing
        module.
    """
    resolved_activation = resolve_activation(activation)
    factory = resolve_graph_conv_factory(
        conv_kind,
        residual=False,
        edge_dim=edge_dim,
        dropout=dropout,
        **conv_kwargs,
    )
    layers = [
        _ActivatedGraphConv(conv=factory(hidden_size, hidden_size), activation=resolved_activation)
        for _ in range(num_layers)
    ]
    return GraphHyperSequential(*layers, num_lanes=num_lanes)


class EmbeddedHyperGraphNetwork(GProjection):
    """GProjection pre-wired with a Hyper-Connection message-passing body.

    Mirrors :class:`~dlkit.domain.nn.graph.embedded.EmbeddedGraphNetwork`'s
    shape exactly (same base class, same ``message_module`` injection
    pattern); the only difference is the message module is a
    :class:`~dlkit.domain.nn.primitives.stacks.GraphHyperSequential` wrapping
    single-layer conv adapters instead of a plain stack.

    The conv sublayer's own residual mechanism is always disabled
    (``residual=False``, fixed, not caller-configurable): the outer
    Hyper-Connection wrapper is what provides the residual path, and the
    lane-mixing matrices are only guaranteed to start at identity when the
    wrapped module itself is non-residual. There is no
    ``ScaledEmbeddedHyperGraphNetwork`` counterpart -- out of scope for this
    class family; it would follow the same ``ScaledGProjection``-basing
    pattern as :class:`~dlkit.domain.nn.graph.embedded.ScaledEmbeddedGraphNetwork`
    if ever needed.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers and message
            module.
        num_layers: Number of stacked graph-conv layers (lanes).
        conv_kind: Graph convolution strategy used for message passing
            (``"gatv2"``, ``"gcn"``, or ``"sage"``).
        num_lanes: Number of Hyper-Connection residual lanes.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            Validity is kind-specific (see :mod:`dlkit.domain.nn.graph.conv`).
        activation: Activation function applied after each conv layer.
        dropout: Dropout probability; semantics are kind-specific.
        message_module: Optional custom message-passing module. When
            supplied, ``conv_kind``/``num_layers``/``num_lanes``/``dropout``/
            ``**conv_kwargs`` are ignored for module construction.
        input_projection: Optional custom input projection module.
        output_projection: Optional custom output projection module.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to the underlying conv.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        num_layers: int,
        conv_kind: GraphConvKind = "gatv2",
        num_lanes: int = 2,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
        **conv_kwargs,
    ) -> None:
        """Initialize EmbeddedHyperGraphNetwork.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers and message
                module.
            num_layers: Number of stacked graph-conv layers (lanes).
            conv_kind: Graph convolution strategy used for message passing.
            num_lanes: Number of Hyper-Connection residual lanes.
            edge_dim: Edge feature dimensionality; ``None`` if no edge
                features.
            activation: Activation function applied after each conv layer.
            dropout: Dropout probability; semantics are kind-specific.
            message_module: Optional custom message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
            **conv_kwargs: Kind-specific knobs forwarded to the underlying
                conv.
        """
        if message_module is None:
            message_module = _build_hyper_message_module(
                conv_kind=conv_kind,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_lanes=num_lanes,
                edge_dim=edge_dim,
                activation=activation,
                dropout=dropout,
                **conv_kwargs,
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            message_module=message_module,
            input_projection=input_projection,
            output_projection=output_projection,
        )


class _GraphMoEBody(nn.Module):
    """Interleaved graph-conv / Sparse-MoE message-passing body.

    Transformer-block style: each layer has two independently residual
    sublayers, a graph-conv sublayer and a Sparse-MoE FFN sublayer. Unlike
    :func:`_build_hyper_message_module`'s conv layers, the conv sublayer here
    is built with ``residual=True`` -- there is no outer Hyper-Connection
    wrapper providing a residual path for it, so it needs its own.

    Args:
        conv_kind: Graph convolution strategy used for every layer's conv
            sublayer.
        hidden_size: Dimension of node feature embeddings.
        num_layers: Number of stacked (conv, MoE) layer pairs.
        num_experts: Number of experts per MoE sublayer.
        top_k: Number of experts routed to per token.
        expert_hidden_features: Internal expansion width used inside each
            expert block, independent of ``hidden_size``. ``None`` keeps the
            block's own default. Only supported by expansion-capable
            ``block_kind`` values.
        block_kind: Dense-block variant used to build each expert.
        linear_kind: Linear-layer variant used inside each expert block.
        router_activation: Router normalization (``"softmax"`` or
            ``"normalized_sigmoid"``).
        capacity_factor: Optional per-expert capacity factor; requires
            ``drop_policy="drop"``.
        drop_policy: Whether over-capacity tokens are dropped.
        jitter_noise: Router logit jitter noise during training.
        edge_dim: Optional edge feature dimensionality.
        activation: Activation function applied after each conv sublayer and
            inside each expert block.
        dropout: Dropout probability; semantics are kind-specific for the
            conv sublayer and expert-block-specific for the MoE sublayer.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to every conv sublayer.
    """

    def __init__(
        self,
        *,
        conv_kind: GraphConvKind,
        hidden_size: int,
        num_layers: int,
        num_experts: int,
        top_k: int = 2,
        expert_hidden_features: int | None = None,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        router_activation: RouterActivation = "softmax",
        capacity_factor: float | None = None,
        drop_policy: DropPolicy = "none",
        jitter_noise: float = 0.0,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        **conv_kwargs,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")

        resolved_activation = resolve_activation(activation)
        conv_factory = resolve_graph_conv_factory(
            conv_kind,
            residual=True,
            edge_dim=edge_dim,
            dropout=dropout,
            **conv_kwargs,
        )
        self.conv_layers = nn.ModuleList(
            [conv_factory(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.moe_layers = nn.ModuleList(
            [
                SparseMoE(
                    in_features=hidden_size,
                    experts=[
                        make_dense_block(
                            block_kind,
                            in_features=hidden_size,
                            out_features=hidden_size,
                            hidden_features=expert_hidden_features,
                            activation=resolved_activation,
                            dropout=dropout,
                            linear_kind=linear_kind,
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
        )
        self.activation = resolved_activation
        self.branch_scale = residual_branch_scale(num_layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None) -> Tensor:
        """Apply interleaved (residual conv, residual MoE) layers sequentially.

        Args:
            x: Node feature tensor of shape ``(num_nodes, hidden_size)``.
            edge_index: Edge indices tensor of shape ``(2, num_edges)``.
            edge_attr: Optional edge features; validity is kind-specific.

        Returns:
            Tensor: Updated node embeddings of shape
                ``(num_nodes, hidden_size)``.
        """
        for conv, moe in zip(self.conv_layers, self.moe_layers, strict=True):
            x = self.activation(conv(x, edge_index, edge_attr))
            moe_out = cast(Tensor, moe(x, return_stats=False))
            x = x + self.branch_scale * moe_out
        return x


class EmbeddedMoEGraphNetwork(GProjection):
    """GProjection pre-wired with an interleaved graph-conv / Sparse-MoE body.

    Unlike :class:`~dlkit.domain.nn.ffnn.hyper_moe.EmbeddedMoEFFNN`, this
    class does **not** support a ``return_stats=True`` tuple-returning
    ``forward()``. ``GProjection``/``ProjectionNetwork.forward()`` has a
    fixed contract that always returns a plain ``Tensor`` (see
    ``ProjectionNetwork._apply_message_module``, which would break if a
    message module ever returned a tuple, since ``self._out_proj(x)`` would
    then be called on a tuple). Widening that shared contract is out of scope
    and risky -- it is relied on by ``GraphHyperConnection``/
    ``GraphHyperSequential`` and every other graph composite. Every
    ``SparseMoE`` layer inside this class's message-passing body is always
    called with ``return_stats=False``, and no ``return_stats`` parameter is
    exposed on this class's constructor or ``forward()``. If routing
    diagnostics are ever needed, reach into the private ``_message_module``
    structure directly (the same convention already used by this package's
    own tests), rather than widening the public ``forward()`` contract.

    The conv sublayer's own residual mechanism is always enabled
    (``residual=True``, fixed, not caller-configurable) -- see
    :class:`_GraphMoEBody` for why.

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        hidden_size: Width of the hidden projection layers and message
            module.
        num_layers: Number of stacked (conv, MoE) layer pairs.
        conv_kind: Graph convolution strategy used for message passing
            (``"gatv2"``, ``"gcn"``, or ``"sage"``).
        num_experts: Number of experts per MoE sublayer.
        top_k: Number of experts routed to per token.
        expert_hidden_features: Internal expansion width used inside each
            expert block, independent of ``hidden_size``.
        block_kind: Dense-block variant used to build each expert.
        linear_kind: Linear-layer variant used inside each expert block.
        router_activation: Router normalization (``"softmax"`` or
            ``"normalized_sigmoid"``).
        capacity_factor: Optional per-expert capacity factor; requires
            ``drop_policy="drop"``.
        drop_policy: Whether over-capacity tokens are dropped.
        jitter_noise: Router logit jitter noise during training.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            Validity is kind-specific (see :mod:`dlkit.domain.nn.graph.conv`).
        activation: Activation function applied after each conv sublayer and
            inside each expert block.
        dropout: Dropout probability; semantics are kind-specific for the
            conv sublayer and expert-block-specific for the MoE sublayer.
        message_module: Optional custom message-passing module. When
            supplied, ``conv_kind``/``num_layers``/``num_experts``/``top_k``/
            ``expert_hidden_features``/``block_kind``/``linear_kind``/
            ``router_activation``/``capacity_factor``/``drop_policy``/
            ``jitter_noise``/``dropout``/``**conv_kwargs`` are ignored for
            module construction.
        input_projection: Optional custom input projection module.
        output_projection: Optional custom output projection module.
        **conv_kwargs: Kind-specific knobs (e.g. ``heads``/``concat`` for
            GATv2, ``aggr`` for SAGE) forwarded to the underlying conv.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_size: int = 64,
        num_layers: int,
        conv_kind: GraphConvKind = "gatv2",
        num_experts: int,
        top_k: int = 2,
        expert_hidden_features: int | None = None,
        block_kind: DenseBlockKind = "parametric",
        linear_kind: DenseLinearKind = "linear",
        router_activation: RouterActivation = "softmax",
        capacity_factor: float | None = None,
        drop_policy: DropPolicy = "none",
        jitter_noise: float = 0.0,
        edge_dim: int | None = None,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0.0,
        message_module: nn.Module | None = None,
        input_projection: nn.Module | None = None,
        output_projection: nn.Module | None = None,
        **conv_kwargs,
    ) -> None:
        """Initialize EmbeddedMoEGraphNetwork.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            hidden_size: Width of the hidden projection layers and message
                module.
            num_layers: Number of stacked (conv, MoE) layer pairs.
            conv_kind: Graph convolution strategy used for message passing.
            num_experts: Number of experts per MoE sublayer.
            top_k: Number of experts routed to per token.
            expert_hidden_features: Internal expansion width used inside each
                expert block.
            block_kind: Dense-block variant used to build each expert.
            linear_kind: Linear-layer variant used inside each expert block.
            router_activation: Router normalization.
            capacity_factor: Optional per-expert capacity factor.
            drop_policy: Whether over-capacity tokens are dropped.
            jitter_noise: Router logit jitter noise during training.
            edge_dim: Edge feature dimensionality; ``None`` if no edge
                features.
            activation: Activation function applied after each conv sublayer
                and inside each expert block.
            dropout: Dropout probability; semantics are kind-specific.
            message_module: Optional custom message-passing module.
            input_projection: Optional custom input projection module.
            output_projection: Optional custom output projection module.
            **conv_kwargs: Kind-specific knobs forwarded to the underlying
                conv.
        """
        if message_module is None:
            message_module = _GraphMoEBody(
                conv_kind=conv_kind,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_experts=num_experts,
                top_k=top_k,
                expert_hidden_features=expert_hidden_features,
                block_kind=block_kind,
                linear_kind=linear_kind,
                router_activation=router_activation,
                capacity_factor=capacity_factor,
                drop_policy=drop_policy,
                jitter_noise=jitter_noise,
                edge_dim=edge_dim,
                activation=activation,
                dropout=dropout,
                **conv_kwargs,
            )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            edge_dim=edge_dim,
            message_module=message_module,
            input_projection=input_projection,
            output_projection=output_projection,
        )
