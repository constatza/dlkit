"""Graph Lightning wrapper for PyTorch Geometric models.

This module provides a Lightning wrapper for graph neural networks that
work with PyTorch Geometric Data objects, using direct processing methods.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Batch, Data

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry

from .base import ProcessingLightningWrapper


class GraphLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for PyTorch Geometric graph neural networks.

    This wrapper handles models that accept PyG Data objects and uses
    direct processing methods from the base wrapper.

    The key specialization is `_forward_features()` which reconstructs
    PyTorch Geometric Data objects from feature dicts before model invocation.

    Example:
        ```python
        wrapper = GraphLightningWrapper(
            settings=wrapper_settings,
            model_settings=model_settings,
            shape_spec=shape_spec,
            entry_configs=data_configs,
        )
        ```
    """

    def __init__(
        self,
        *,
        settings: WrapperComponentSettings,
        model_settings: ModelComponentSettings,
        entry_configs: dict[str, DataEntry] | None = None,
        **kwargs,
    ):
        """Initialize the graph Lightning wrapper.

        Args:
            settings: Wrapper configuration settings
            model_settings: Model configuration settings
            entry_configs: Data entry configurations
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            **kwargs,
        )

    def forward(self, data: Data | Batch) -> Tensor:
        """Forward pass through the model with PyG Data input.

        Args:
            data: PyTorch Geometric Data or Batch object containing x, edge_index, etc.
                  Batch objects are created by PyG DataLoader for batched graphs.

        Returns:
            Model output tensor
        """
        return self.model(data)

    def _forward_features(self, features: dict[str, Tensor] | Tensor) -> dict[str, Tensor] | Tensor:
        """Override to reconstruct PyG Data object from feature dict.

        Graph models expect PyG Data objects, not raw tensors or dicts.
        Always reconstruct Data fresh from dict/tensor for serialization.

        Args:
            features: Feature dict or tensor from batch extraction

        Returns:
            Model output
        """
        # Use match-case for clean type handling
        match features:
            case dict():
                # Reconstruct Data object from feature dict
                # Filter to only tensor values for serialization safety
                data_kwargs = {
                    key: value
                    for key, value in features.items()
                    if isinstance(value, Tensor)
                }
                data = Data(**data_kwargs)
                return self.model(data)

            case Tensor():
                # Wrap bare tensor in Data object
                data = Data(x=features)
                return self.model(data)

            case _:
                # Fallback for unexpected types
                raise TypeError(f"Unexpected feature type: {type(features)}")
