"""Deep Operator Network (DeepONet) — composable and convenience variants.

Reference: Lu et al., "Learning nonlinear operators via DeepONet based
on the universal approximation theorem of operators",
Nature Machine Intelligence 3, 218–229 (2021).
https://doi.org/10.1038/s42256-021-00302-5

Two classes are provided:

``DeepONet``
    Pure composition: accepts pre-built ``branch_net`` and ``trunk_net``
    modules.  Use this when you want to inject any architecture (transformer,
    CNN, custom FFNN) as the branch or trunk.

``MLPDeepONet``
    Convenience subclass that builds ``FeedForwardNN`` branch and trunk
    networks from scalar hyperparameters.  Shape injection by the factory
    "ffnn" strategy works automatically (``in_features`` is accepted).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dlkit.domain.nn.ffnn.simple import FeedForwardNN


class DeepONet(nn.Module):
    """Deep Operator Network with injectable branch and trunk networks.

    DeepONet approximates an operator 𝒢 : U → V by decomposing it into:

    * **Branch net**: encodes the input function ``u`` (sensor readings) into
      a ``trunk_width * out_features``-dimensional embedding.
    * **Trunk net**: encodes each query coordinate ``y`` into a matching
      embedding.
    * The dot product of the two embeddings, reshaped and biased, gives the
      output value at each query point.

    This class accepts *pre-built* branch and trunk modules.  Their output
    sizes must both equal ``trunk_width * out_features``.  Use ``MLPDeepONet``
    if you want the sub-networks built automatically from scalar parameters.

    Args:
        branch_net: Encodes the input function.
            Input shape: ``(batch, n_sensors)``.
            Output shape: ``(batch, trunk_width * out_features)``.
        trunk_net: Encodes query coordinates.
            Input shape: ``(n_queries, n_coords)`` or per-batch.
            Output shape: ``(*query_shape, trunk_width * out_features)``.
        trunk_width: Size of the dot-product latent space per output channel.
        out_features: Number of output values per query point.

    Example — inject a transformer as the branch::

        branch = TransformerEncoder(...)  # (B, n_sensors) → (B, p)
        trunk = FeedForwardNN(...)  # (B, n_coords)  → (B, p)
        model = DeepONet(
            branch_net=branch,
            trunk_net=trunk,
            trunk_width=64,
            out_features=1,
        )
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

    @property
    def out_features(self) -> int:
        """Number of output values per query point.

        Returns:
            ``out_features`` passed at construction.
        """
        return self._out_features

    def forward(self, u: Tensor, y: Tensor) -> Tensor:
        """Evaluate the operator at the given query points.

        Args:
            u: Input function values at fixed sensor locations,
               shape ``(batch, n_sensors)``.
            y: Query coordinates, shape
               ``(batch, n_queries, n_coords)`` or
               ``(n_queries, n_coords)`` (broadcast over batch).

        Returns:
            Output function values of shape
            ``(batch, n_queries, out_features)``.
        """
        batch = u.shape[0]

        # Branch embedding: (B, p)
        b = self.branch_net(u)

        # Trunk embedding over query points
        if y.dim() == 2:
            y = y.unsqueeze(0).expand(batch, -1, -1)
        n_queries = y.shape[1]
        y_flat = y.reshape(batch * n_queries, -1)
        t = self.trunk_net(y_flat)
        t = t.reshape(batch, n_queries, -1)

        # Reshape to (B, out_features, trunk_width) for dot product
        b = b.reshape(batch, self._out_features, self._trunk_width)
        t = t.reshape(batch, n_queries, self._out_features, self._trunk_width)

        # Dot product → (B, Q, out_features)
        v = torch.einsum("bop,bqop->bqo", b, t) / self._trunk_width
        return v + self.bias


class MLPDeepONet(DeepONet):
    """DeepONet with ``FeedForwardNN`` branch and trunk networks.

    Convenience subclass that constructs branch and trunk as
    ``FeedForwardNN`` instances from scalar hyperparameters.  The factory
    "ffnn" strategy injects ``in_features`` automatically from the dataset
    shape summary.

    Args:
        in_features: Number of sensor locations (branch input size).
            Injected from the dataset shape summary by the factory.
        out_features: Number of output values per query point.
        n_coords: Spatial coordinate dimension for trunk query points.
        trunk_width: Size of the dot-product latent space per output channel.
        branch_depth: Number of hidden layers in the branch net.
        trunk_depth: Number of hidden layers in the trunk net.
        hidden_size: Hidden layer width for both sub-networks.
        activation: Pointwise activation for both sub-networks.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        n_coords: int,
        trunk_width: int = 64,
        branch_depth: int = 4,
        trunk_depth: int = 4,
        hidden_size: int = 128,
        activation: Callable[[Tensor], Tensor] = F.gelu,
    ) -> None:
        latent_dim = trunk_width * out_features
        branch_net = FeedForwardNN(
            in_features=in_features,
            out_features=latent_dim,
            layers=[hidden_size] * branch_depth,
            activation=activation,
        )
        trunk_net = FeedForwardNN(
            in_features=n_coords,
            out_features=latent_dim,
            layers=[hidden_size] * trunk_depth,
            activation=activation,
        )
        super().__init__(
            branch_net=branch_net,
            trunk_net=trunk_net,
            trunk_width=trunk_width,
            out_features=out_features,
        )
