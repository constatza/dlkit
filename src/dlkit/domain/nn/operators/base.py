"""Base classes and protocols for neural operator architectures.

Hierarchy
---------
``IOperatorNetwork``
    Capability marker: the model approximates an operator between function spaces.

  ``IGridOperator``   (extends ``IOperatorNetwork``)
    Operator on fixed spatial grids — ``forward(x) → y``.

  ``IQueryOperator``  (extends ``IOperatorNetwork``)
    Operator that maps an input function to arbitrary query locations —
    ``forward(u, y) → v``.

Concrete base classes
---------------------
``GridOperatorBase``
    Lifting → body → projection scaffold for grid-to-grid operators.
    Inject any ``nn.Module`` body to build a new operator family without
    modifying existing code (open-closed principle).

The split at ``IGridOperator`` / ``IQueryOperator`` reflects the genuine
difference in forward-pass signature.  Merging them into one protocol
would force either interface to accept arguments it doesn't use (LSP
violation) or require awkward keyword-only optional arguments.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor, nn


@runtime_checkable
class IOperatorNetwork(Protocol):
    """Marker protocol: network that approximates an operator between function spaces.

    Use this protocol for capability detection only (e.g.
    ``isinstance(model, IOperatorNetwork)``).  For invocation, use the
    more specific ``IGridOperator`` or ``IQueryOperator``.
    """

    @property
    def out_features(self) -> int:
        """Primary output feature / channel count.

        Returns:
            Number of output channels or features.
        """
        ...


@runtime_checkable
class IGridOperator(IOperatorNetwork, Protocol):
    """Operator on a fixed spatial grid.

    The input and output share the same spatial resolution.
    ``forward`` accepts a single batched tensor.

    Typical input shape:  ``(batch, in_channels, length)``
    Typical output shape: ``(batch, out_channels, length)``
    """

    def forward(self, x: Tensor) -> Tensor:
        """Apply the grid operator.

        Args:
            x: Input function discretised on a regular grid,
               shape ``(batch, in_channels, length)``.

        Returns:
            Output function on the same grid,
            shape ``(batch, out_channels, length)``.
        """
        ...


@runtime_checkable
class IQueryOperator(IOperatorNetwork, Protocol):
    """Operator mapping an input function to arbitrary query locations.

    The input function is provided as a vector of sensor readings ``u``.
    The output is evaluated at the query coordinates ``y``.

    Typical input shapes:
        ``u``: ``(batch, n_sensors)``
        ``y``: ``(batch, n_queries, n_coords)`` or ``(n_queries, n_coords)``

    Typical output shape: ``(batch, n_queries, out_features)``
    """

    def forward(self, u: Tensor, y: Tensor) -> Tensor:
        """Apply the query operator.

        Args:
            u: Input function values at fixed sensor locations,
               shape ``(batch, n_sensors)``.
            y: Query coordinates at which to evaluate the output,
               shape ``(batch, n_queries, n_coords)`` or
               ``(n_queries, n_coords)`` (broadcast over batch).

        Returns:
            Output function values at the query points,
            shape ``(batch, n_queries, out_features)``.
        """
        ...


class GridOperatorBase(nn.Module):
    """Scaffold for grid-to-grid neural operators.

    Encapsulates the standard three-stage pipeline used by grid operators:

    1. **Lifting** — pointwise Conv1d raises ``in_channels`` to the latent
       ``width``.
    2. **Body** — any ``nn.Module`` that maps
       ``(batch, width, length) → (batch, width, length)``.  Inject different
       body implementations to produce different operator families without
       modifying this class (open-closed principle).
    3. **Projection** — pointwise Conv1d maps ``width`` to ``out_channels``.

    Args:
        body: Core operator body. Must accept and return tensors of shape
            ``(batch, width, length)``.
        in_channels: Number of input function channels.
        out_channels: Number of output function channels.
        width: Latent channel width used throughout the body.

    Example — custom operator from a novel body::

        class WaveletNO(GridOperatorBase):
            def __init__(self, *, in_channels, out_channels, width, levels):
                body = nn.Sequential(
                    *[WaveletLayer(channels=width, levels=levels) for _ in range(4)]
                )
                super().__init__(
                    body=body,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    width=width,
                )
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
        """Number of output channels.

        Returns:
            ``out_channels`` passed at construction.
        """
        return self._out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Apply lifting → body → projection.

        Args:
            x: Input function of shape ``(batch, in_channels, length)``.

        Returns:
            Output function of shape ``(batch, out_channels, length)``.
        """
        return self.projection(self.body(self.lifting(x)))
