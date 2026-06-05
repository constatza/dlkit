"""Base classes and protocols for neural operator architectures."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class IOperatorNetwork(Protocol):
    """Marker protocol for operator networks."""

    @property
    def out_features(self) -> int:
        """Primary output feature or channel count."""
        ...


@runtime_checkable
class IGridOperator(IOperatorNetwork, Protocol):
    """Operator on a fixed spatial grid.

    Input/output dimensions:
        input: ``(batch, in_channels, *spatial_shape)``
        output: ``(batch, out_channels, *spatial_shape)``
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply the operator.

        Input/output dimensions:
            input: ``(batch, in_channels, *spatial_shape)``
            output: ``(batch, out_channels, *spatial_shape)``
        """
        ...


@runtime_checkable
class IQueryOperator(IOperatorNetwork, Protocol):
    """Operator evaluated at query coordinates.

    Input/output dimensions:
        branch input: ``(batch, *branch_shape)``
        query input: ``(batch, n_queries, query_dim)``
        output: ``(batch, n_queries, out_features)``
    """

    def forward(self, u: Tensor, y: Tensor) -> Tensor:
        """Apply the operator.

        Input/output dimensions:
            branch input: ``(batch, *branch_shape)``
            query input: ``(batch, n_queries, query_dim)``
            output: ``(batch, n_queries, out_features)``
        """
        ...


class GridOperatorBase(nn.Module):
    """Scaffold for grid-to-grid operators.

    Input/output dimensions:
        input: ``(batch, in_channels, *spatial_shape)``
        output: ``(batch, out_channels, *spatial_shape)``

    Internal dimensions:
        body input: ``(batch, width, *spatial_shape)``
        body output: ``(batch, width, *spatial_shape)``
    """

    def __init__(
        self,
        *,
        body: nn.Module,
        in_channels: int,
        out_channels: int,
        width: int,
    ) -> None:
        super().__init__()
        self.lifting = nn.Conv1d(in_channels, width, kernel_size=1)
        self.body = body
        self.projection = nn.Conv1d(width, out_channels, kernel_size=1)
        self._out_channels = out_channels

    @property
    def out_features(self) -> int:
        """Number of output channels."""
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply lifting, body, and projection.

        Input/output dimensions:
            input: ``(batch, in_channels, *spatial_shape)``
            output: ``(batch, out_channels, *spatial_shape)``
        """
        return self.projection(self.body(self.lifting(x)))
