from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

from torch import Tensor, nn

if TYPE_CHECKING:
    from dlkit.common.shapes import ShapeContext


class BaseGraphNetwork(nn.Module):
    """Abstract base for graph neural networks (PyG-based).

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
        **kwargs: Additional model-specific parameters.
    """

    _SHAPE_KWARG_NAMES: frozenset[str] = frozenset({"in_channels", "out_channels"})

    def __init__(
        self, *, in_channels: int, out_channels: int, edge_dim: int | None = None, **kwargs
    ):
        """Initialize BaseGraphNetwork with explicit channel dimensions.

        Args:
            in_channels: Number of input node feature channels.
            out_channels: Number of output node feature channels.
            edge_dim: Edge feature dimensionality; ``None`` if no edge features.
            **kwargs: Additional parameters passed to subclasses.
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._edge_dim = edge_dim

    @classmethod
    def shape_kwarg_names(cls) -> frozenset[str]:
        """Return the set of constructor kwargs derived from shapes.

        Returns:
            Frozenset of kwarg names supplied by ``from_context``.
        """
        return cls._SHAPE_KWARG_NAMES

    @classmethod
    def from_context(cls, context: ShapeContext, **kwargs: Any) -> Self:
        """Construct the model from a ShapeContext.

        Node and output channels are read from the last dimension of the first
        input and output shapes respectively. An optional ``edge_dim`` may be
        supplied via ``kwargs``.

        Args:
            context: Shape context with input and output shape mappings.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance of this model.
        """
        node_shape = next(iter(context.input_shapes.values()))
        out_shape = next(iter(context.output_shapes.values()))
        in_channels = node_shape[-1]
        out_channels = out_shape[-1]
        edge_dim = kwargs.pop("edge_dim", None)
        return cls(in_channels=in_channels, out_channels=out_channels, edge_dim=edge_dim, **kwargs)

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with decomposed graph tensors.

        Args:
            x: Node feature tensor.
            edge_index: Edge connectivity tensor (2 × num_edges).
            edge_attr: Optional edge attribute tensor.

        Returns:
            Output tensor from graph processing.
        """
        ...
