from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

from torch import Tensor, nn

if TYPE_CHECKING:
    from dlkit.domain.nn.contracts import ModelContractSpec


class BaseGraphNetwork(nn.Module):
    """Abstract base for graph neural networks (PyG-based).

    Args:
        in_channels: Number of input node feature channels.
        out_channels: Number of output node feature channels.
        edge_dim: Edge feature dimensionality; ``None`` if no edge features.
        **kwargs: Additional model-specific parameters.
    """

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
    def from_contract(cls, contract: ModelContractSpec, **kwargs: Any) -> Self:
        """Construct the model from a ``GraphContractSpec``.

        Args:
            contract: A ``ModelContractSpec`` that must be a ``GraphContractSpec``.
            **kwargs: Additional keyword arguments forwarded to the constructor.

        Returns:
            A fully constructed instance of this model.

        Raises:
            TypeError: If ``contract`` is not a ``GraphContractSpec``.
        """
        from dlkit.domain.nn.contracts import GraphContractSpec

        match contract:
            case GraphContractSpec(in_channels=c_in, out_channels=c_out, edge_dim=e_dim):
                return cls(in_channels=c_in, out_channels=c_out, edge_dim=e_dim, **kwargs)
            case _:
                raise TypeError(
                    f"{cls.__name__} requires GraphContractSpec, got {type(contract).__name__}"
                )

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
