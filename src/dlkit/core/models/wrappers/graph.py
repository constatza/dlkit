"""Graph Lightning wrapper for PyTorch Geometric models.

This module provides a Lightning wrapper for graph neural networks that
work with PyTorch Geometric Data objects, using direct processing methods.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor
from torch_geometric.data import Batch as PyGBatch, Data

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry, is_target_entry

from .base import ProcessingLightningWrapper


class GraphLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for PyTorch Geometric graph neural networks.

    Accepts PyG ``Data``/``Batch`` objects from a PyG DataLoader directly —
    no dlkit positional Batch conversion needed. Target is extracted by the
    first target-entry config name (default ``"y"``). Entry names appear only
    at construction time; they never flow through the data path.

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
        entry_configs: tuple[DataEntry, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the graph Lightning wrapper.

        Args:
            settings: Wrapper configuration settings.
            model_settings: Model configuration settings.
            entry_configs: Data entry configurations.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            **kwargs,
        )
        # Resolved once at init from config; never referenced in the data hot path
        self._graph_target_name: str = next(
            (e.name for e in self._entry_configs if is_target_entry(e)),
            "y",
        )

    @staticmethod
    def _decompose_pyg_batch(
        batch: Data | PyGBatch,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Extract (x, edge_index, edge_attr) from a PyG Data/Batch.

        Args:
            batch: PyG Data or Batch object to decompose.

        Returns:
            Tuple of (x, edge_index, edge_attr) tensors.
        """
        x: Tensor = batch.x
        edge_index: Tensor = batch.edge_index
        edge_attr: Tensor | None = getattr(batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(batch, "edge_weight", None)
        return x, edge_index, edge_attr

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through the model with decomposed graph tensors.

        Args:
            x: Node feature tensor.
            edge_index: Edge connectivity tensor (2 × num_edges).
            edge_attr: Optional edge attribute tensor.

        Returns:
            Model output tensor.
        """
        return self.model(x, edge_index, edge_attr)

    # =========================================================================
    # Step overrides: PyG Data/Batch → model directly
    # =========================================================================

    def training_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Training step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the training loss.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        loss = self.loss_function(predictions, target)
        self._log_stage_outputs("train", loss)
        return {"loss": loss}

    def validation_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Validation step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the validation loss.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        val_loss = self.loss_function(predictions, target)
        metrics = self._update_metrics(predictions, (target,), "val")
        self._log_stage_outputs("val", val_loss, metrics)
        return {"val_loss": val_loss}

    def test_step(self, batch: Data | PyGBatch, batch_idx: int) -> dict[str, Any]:
        """Test step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Dictionary containing the test loss.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        target = self._extract_pyg_target(batch, predictions)
        test_loss = self.loss_function(predictions, target)
        metrics = self._update_metrics(predictions, (target,), "test")
        self._log_stage_outputs("test", test_loss, metrics)
        return {"test_loss": test_loss}

    def predict_step(self, batch: Data | PyGBatch, batch_idx: int) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Predict step for PyG Data/Batch.

        Args:
            batch: PyG Data or Batch from a PyG DataLoader.
            batch_idx: Index of the batch.

        Returns:
            Tuple of (predictions, targets, latents), each containing a tuple of tensors.
        """
        x, edge_index, edge_attr = self._decompose_pyg_batch(batch)
        predictions = self.model(x, edge_index, edge_attr)
        raw_target = getattr(batch, self._graph_target_name, None)
        targets: tuple[Tensor, ...] = (raw_target,) if raw_target is not None else ()
        return (
            (predictions,) if isinstance(predictions, Tensor) else predictions,
            targets,
            (),
        )

    def _extract_pyg_target(self, batch: Data | PyGBatch, predictions: Tensor) -> Tensor:
        """Extract target from PyG batch and align dtype to predictions.

        Args:
            batch: PyG Data or Batch with a target attribute.
            predictions: Model predictions tensor (used for dtype alignment).

        Returns:
            Target tensor with dtype matching predictions.

        Raises:
            AttributeError: If the target attribute is not found on the batch.
        """
        target: Tensor = getattr(batch, self._graph_target_name)
        if target.is_floating_point() and target.dtype != predictions.dtype:
            target = target.to(dtype=predictions.dtype)
        return target
