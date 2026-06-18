"""Deep Operator Network (DeepONet) variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Literal, Self, cast

import torch
from torch import Tensor, nn

from dlkit.common.sources import InputShapes, OutputShapes
from dlkit.domain.nn.ffnn.residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation


class DeepONet(nn.Module):
    """Deep Operator Network with injectable branch and trunk networks.

    Input/output dimensions:
        branch input: ``(batch, *branch_shape)``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch_net output: ``(batch, trunk_width * out_features)``
        trunk_net output: ``(batch * n_queries, trunk_width * out_features)``

    Constructor dimensions:
        ``trunk_width``, ``out_features``
    """

    def __init__(
        self,
        *,
        branch_net: nn.Module,
        trunk_net: nn.Module,
        trunk_width: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self._trunk_width = trunk_width
        self._out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("_trunk_scale", torch.sqrt(torch.tensor(float(trunk_width))))

    @property
    def out_features(self) -> int:
        """Number of output values per query point."""
        return self._out_features

    def forward(self, u: Tensor, y: Tensor) -> Tensor:
        """Evaluate the operator.

        Input/output dimensions:
            branch input: ``(batch, *branch_shape)``
            query input: ``(batch, n_queries, query_dim)``
            output: ``(batch, n_queries, out_features)``
        """
        if y.dim() != 3:
            raise ValueError(
                f"DeepONet trunk input must have canonical shape (B, Q, d); got {tuple(y.shape)}"
            )
        batch = u.shape[0]
        if y.shape[0] != batch:
            raise ValueError(
                "DeepONet branch and trunk batch dimensions must match; "
                f"got u.shape[0]={batch} and y.shape[0]={y.shape[0]}"
            )

        expected_width = self._out_features * self._trunk_width

        branch = self.branch_net(u)
        if branch.shape != (batch, expected_width):
            raise ValueError(
                "branch_net must return shape "
                f"{(batch, expected_width)} for DeepONet; got {tuple(branch.shape)}"
            )
        branch = branch.reshape(batch, self._out_features, self._trunk_width)

        y_flat = y.reshape(batch * y.shape[1], -1)
        trunk = self.trunk_net(y_flat)
        n_queries = y.shape[1]
        if trunk.shape != (batch * n_queries, expected_width):
            raise ValueError(
                "trunk_net must return shape "
                f"{(batch * n_queries, expected_width)} for DeepONet; got {tuple(trunk.shape)}"
            )
        trunk = trunk.reshape(batch, n_queries, self._out_features, self._trunk_width)
        trunk_scale = cast(Tensor, self._trunk_scale)
        values: Tensor = (
            torch.einsum("bop,bqop->bqo", branch, trunk) / trunk_scale
        ) + self.bias.view(1, 1, -1)
        return values


class _FlatBranchDeepONet(DeepONet):
    """Internal DeepONet base for flattened branch inputs.

    Input/output dimensions:
        branch input: ``(batch, *branch_shape)``
        flattened branch input: ``(batch, prod(branch_shape))``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``
    """

    def forward(self, u: Tensor, y: Tensor) -> Tensor:
        """Flatten the branch input before dispatch."""
        return super().forward(u.reshape(u.shape[0], -1), y)


class VarWidthDeepONet(_FlatBranchDeepONet):
    """DeepONet with variable-width FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, trunk_width * out_features)``
        trunk FFNN output: ``(batch * n_queries, trunk_width * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``query_dim = query_shape[-1]`` derived from the query input shape
        ``trunk_width``, ``out_features``, ``branch_layers``, ``trunk_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        query_dim: int,
        trunk_width: int = 64,
        branch_layers: Sequence[int],
        trunk_layers: Sequence[int],
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        if not branch_layers:
            raise ValueError("branch_layers must contain at least one hidden width")
        if not trunk_layers:
            raise ValueError("trunk_layers must contain at least one hidden width")
        resolved = resolve_activation(activation)
        latent_dim = trunk_width * out_features
        branch_net = VarWidthFFNN(
            in_features=branch_in_features,
            out_features=latent_dim,
            layers=branch_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        trunk_net = VarWidthFFNN(
            in_features=query_dim,
            out_features=latent_dim,
            layers=trunk_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            trunk_width=trunk_width,
            out_features=out_features,
        )

    @classmethod
    def from_entries(
        cls, input_shapes: InputShapes, output_shapes: OutputShapes, **kwargs: Any
    ) -> Self:
        """Build the operator from dataset entry shapes.

        The first input is the branch (sensor) function and the second is the
        query coordinate grid. When only one input is provided it serves as both.

        Args:
            input_shapes: Mapping from input name to its per-sample shape.
            output_shapes: Mapping from output name to its per-sample shape.
            **kwargs: Additional constructor arguments.

        Returns:
            Constructed DeepONet variant instance.
        """
        shapes = list(input_shapes.values())
        branch_shape = shapes[0]
        query_shape = shapes[1] if len(shapes) > 1 else shapes[0]
        out_features = next(iter(output_shapes.values()))[0]
        return cls(
            branch_in_features=prod(branch_shape),
            query_dim=query_shape[-1],
            out_features=out_features,
            **kwargs,
        )


class FFNNDeepONet(_FlatBranchDeepONet):
    """DeepONet with constant-width residual FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, trunk_width * out_features)``
        trunk FFNN output: ``(batch * n_queries, trunk_width * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``query_dim = query_shape[-1]`` derived from the query input shape
        ``trunk_width``, ``out_features``, ``branch_hidden_size``,
        ``branch_num_layers``, ``trunk_hidden_size``, ``trunk_num_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        query_dim: int,
        trunk_width: int = 64,
        branch_hidden_size: int | None = None,
        branch_num_layers: int = 4,
        trunk_hidden_size: int | None = None,
        trunk_num_layers: int = 4,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        resolved = resolve_activation(activation)
        latent_dim = trunk_width * out_features
        branch_net = FFNN(
            in_features=branch_in_features,
            out_features=latent_dim,
            hidden_size=branch_hidden_size,
            num_layers=branch_num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        trunk_net = FFNN(
            in_features=query_dim,
            out_features=latent_dim,
            hidden_size=trunk_hidden_size,
            num_layers=trunk_num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            trunk_width=trunk_width,
            out_features=out_features,
        )


class EmbeddedDeepONet(_FlatBranchDeepONet):
    """DeepONet with embedded dense residual FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, trunk_width * out_features)``
        trunk FFNN output: ``(batch * n_queries, trunk_width * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``query_dim = query_shape[-1]`` derived from the query input shape
        ``trunk_width``, ``out_features``, ``branch_hidden_size``,
        ``branch_num_layers``, ``trunk_hidden_size``, ``trunk_num_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        query_dim: int,
        trunk_width: int = 64,
        branch_hidden_size: int | None = None,
        branch_num_layers: int = 4,
        trunk_hidden_size: int | None = None,
        trunk_num_layers: int = 4,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        resolved = resolve_activation(activation)
        latent_dim = trunk_width * out_features
        branch_net = EmbeddedFFNN(
            in_features=branch_in_features,
            out_features=latent_dim,
            hidden_size=branch_hidden_size,
            num_layers=branch_num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        trunk_net = EmbeddedFFNN(
            in_features=query_dim,
            out_features=latent_dim,
            hidden_size=trunk_hidden_size,
            num_layers=trunk_num_layers,
            activation=resolved,
            normalize=normalize,
            dropout=dropout,
            bias=bias,
        )
        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            trunk_width=trunk_width,
            out_features=out_features,
        )
