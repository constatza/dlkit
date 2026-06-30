"""Deep Operator Network (DeepONet) variants."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import prod
from typing import Literal, cast

import torch
from torch import Tensor, nn

from dlkit.common.shapes import ShapeContext
from dlkit.domain.nn.contracts import (
    InputSpec as _InputSpec,
)
from dlkit.domain.nn.contracts import StandardEntryConsumer
from dlkit.domain.nn.ffnn.residual import FFNN, EmbeddedFFNN, VarWidthFFNN
from dlkit.domain.nn.types import ActivationName
from dlkit.domain.nn.utils import resolve_activation


class DeepONet(nn.Module):
    """Deep Operator Network with injectable branch and trunk networks.

    Input/output dimensions:
        branch input: ``(batch, *branch_shape)``
        trunk input: ``(batch, n_queries, trunk_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch_net output: ``(batch, basis_dim * out_features)``
        trunk_net output: ``(batch * n_queries, basis_dim * out_features)``

    Constructor dimensions:
        ``basis_dim``, ``out_features``
    """

    def __init__(
        self,
        *,
        branch_net: nn.Module,
        trunk_net: nn.Module,
        basis_dim: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self._basis_dim = basis_dim
        self._out_features = out_features
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("_basis_scale", torch.sqrt(torch.tensor(float(basis_dim))))

    @property
    def out_features(self) -> int:
        """Number of output values per query point."""
        return self._out_features

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        """Evaluate the operator.

        Input/output dimensions:
            branch input: ``(batch, *branch_shape)``
            trunk input: ``(batch, n_queries, trunk_dim)``
            output: ``(batch, n_queries, out_features)``
        """
        if trunk.dim() != 3:
            raise ValueError(
                "DeepONet trunk input must have canonical shape "
                f"(B, Q, d); got {tuple(trunk.shape)}"
            )
        batch = branch.shape[0]
        if trunk.shape[0] != batch:
            raise ValueError(
                "DeepONet branch and trunk batch dimensions must match; "
                f"got branch.shape[0]={batch} and trunk.shape[0]={trunk.shape[0]}"
            )

        expected_width = self._out_features * self._basis_dim

        branch_values = self.branch_net(branch)
        if branch_values.shape != (batch, expected_width):
            raise ValueError(
                "branch_net must return shape "
                f"{(batch, expected_width)} for DeepONet; got {tuple(branch_values.shape)}"
            )
        branch_values = branch_values.reshape(batch, self._out_features, self._basis_dim)

        trunk_flat = trunk.reshape(batch * trunk.shape[1], -1)
        trunk_values = self.trunk_net(trunk_flat)
        n_queries = trunk.shape[1]
        if trunk_values.shape != (batch * n_queries, expected_width):
            raise ValueError(
                "trunk_net must return shape "
                f"{(batch * n_queries, expected_width)} for DeepONet; got {tuple(trunk_values.shape)}"
            )
        trunk_values = trunk_values.reshape(batch, n_queries, self._out_features, self._basis_dim)
        trunk_scale = cast(Tensor, self._basis_scale)
        values: Tensor = (
            torch.einsum("bop,bqop->bqo", branch_values, trunk_values) / trunk_scale
        ) + self.bias.view(1, 1, -1)
        return values


class _FlatBranchDeepONet(StandardEntryConsumer, DeepONet):
    """Internal DeepONet base for flattened branch inputs.

    Input/output dimensions:
        branch input: ``(batch, *branch_shape)``
        flattened branch input: ``(batch, prod(branch_shape))``
        trunk input: ``(batch, n_queries, trunk_dim)``
        output: ``(batch, n_queries, out_features)``

    Expected entry names: ``"branch"`` for sensor readings, ``"trunk"`` for
    coordinate points. When only one entry is provided it serves as both.
    """

    class InputSpec(_InputSpec):
        branch: tuple[int, ...]  # sensor readings — flat or multi-dimensional
        trunk: tuple[int, ...]  # coordinate points, last dim = trunk_dim

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset(
        {"branch_in_features", "trunk_dim", "out_features"}
    )

    @classmethod
    def resolve_shape_kwargs(cls, context: ShapeContext) -> dict[str, int]:
        """Derive branch/trunk/output dimensions from entry shapes.

        Uses named lookup via InputSpec (``"branch"`` / ``"trunk"``).

        Args:
            context: Shape context carrying input and output shapes.

        Returns:
            Dict with ``branch_in_features``, ``trunk_dim``, and ``out_features``.

        Raises:
            ValueError: If the trunk shape has fewer than 1 dimension.
        """
        branch_shape = context.input_shapes["branch"]
        trunk_shape = context.input_shapes["trunk"]
        if len(trunk_shape) < 1:
            raise ValueError(
                f"{cls.__name__} requires at least 1-D trunk shape but got {trunk_shape}"
            )
        return {
            "branch_in_features": prod(branch_shape),
            "trunk_dim": trunk_shape[-1],
            "out_features": next(iter(context.output_shapes.values()))[-1],
        }

    def forward(self, branch: Tensor, trunk: Tensor) -> Tensor:
        """Flatten the branch input before dispatch."""
        return super().forward(branch.reshape(branch.shape[0], -1), trunk)


class VarWidthDeepONet(_FlatBranchDeepONet):
    """DeepONet with variable-width FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        trunk input: ``(batch, n_queries, trunk_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, basis_dim * out_features)``
        trunk FFNN output: ``(batch * n_queries, basis_dim * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``trunk_dim = trunk_shape[-1]`` derived from the trunk input shape
        ``basis_dim``, ``out_features``, ``branch_layers``, ``trunk_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        trunk_dim: int,
        basis_dim: int,
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
        latent_dim = basis_dim * out_features
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
            in_features=trunk_dim,
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
            basis_dim=basis_dim,
            out_features=out_features,
        )


class FFNNDeepONet(_FlatBranchDeepONet):
    """DeepONet with constant-width residual FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        trunk input: ``(batch, n_queries, trunk_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, basis_dim * out_features)``
        trunk FFNN output: ``(batch * n_queries, basis_dim * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``trunk_dim = trunk_shape[-1]`` derived from the trunk input shape
        ``basis_dim``, ``out_features``, ``branch_hidden_size``,
        ``branch_num_layers``, ``trunk_hidden_size``, ``trunk_num_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        trunk_dim: int,
        basis_dim: int,
        branch_hidden_size: int,
        branch_num_layers: int = 4,
        trunk_hidden_size: int,
        trunk_num_layers: int = 4,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        resolved = resolve_activation(activation)
        latent_dim = basis_dim * out_features
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
            in_features=trunk_dim,
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
            basis_dim=basis_dim,
            out_features=out_features,
        )


class EmbeddedDeepONet(_FlatBranchDeepONet):
    """DeepONet with embedded dense residual FFNN branch and trunk networks.

    Input/output dimensions:
        branch input after flattening: ``(batch, flattened_branch_width)``
        trunk input: ``(batch, n_queries, trunk_dim)``
        output: ``(batch, n_queries, out_features)``

    Architecture dimensions:
        branch FFNN output: ``(batch, basis_dim * out_features)``
        trunk FFNN output: ``(batch * n_queries, basis_dim * out_features)``

    Constructor dimensions:
        ``branch_in_features``: flattened branch width
        ``branch_in_features = prod(branch_shape)`` derived from the first input shape
        common sensor-vector case: ``branch_shape = (n_sensors,)`` gives
        ``branch_in_features = n_sensors``
        ``trunk_dim = trunk_shape[-1]`` derived from the trunk input shape
        ``basis_dim``, ``out_features``, ``branch_hidden_size``,
        ``branch_num_layers``, ``trunk_hidden_size``, ``trunk_num_layers``
    """

    def __init__(
        self,
        *,
        branch_in_features: int,
        out_features: int,
        trunk_dim: int,
        basis_dim: int,
        branch_hidden_size: int,
        branch_num_layers: int = 4,
        trunk_hidden_size: int,
        trunk_num_layers: int = 4,
        activation: ActivationName | Callable[[Tensor], Tensor] | None = None,
        normalize: Literal["batch", "layer"] | None = None,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        resolved = resolve_activation(activation)
        latent_dim = basis_dim * out_features
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
            in_features=trunk_dim,
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
            basis_dim=basis_dim,
            out_features=out_features,
        )
