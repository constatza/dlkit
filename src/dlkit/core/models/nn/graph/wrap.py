import torch
from torch_geometric.data import Data

from dlkit.core.models.nn import BaseWrapper


class GraphNetwork(BaseWrapper):
    """
    A LightningModule wrapper for PyTorch Geometric networks.

    This class handles the training, validation, testing, and prediction
    loops for graph-based models built with PyTorch Geometric.
    Pure functions and model components are separated for clarity and testability.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Perform a forward pass on a PyG Data object.

        Args:
            data: A torch_geometric.Data instance containing x, edge_index, etc.
        Returns:
            Model predictions (raw tensor).
        """
        return self.model(data)

    def training_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Training step for a batch of graph

        Args:
            batch: A Data object containing x, edge_index, and targets.
            batch_idx: Index of the batch.
        Returns:
            A dict containing the training loss.
        """
        # Forward pass
        out = self.forward(batch)
        # Extract ground truth
        y_true = batch.y
        # Compute loss
        loss = self.loss_function(out, y_true)
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Validation step for a batch of graph
        """
        out = self.forward(batch)
        y_true = batch.y
        val_loss = self.loss_function(out, y_true)
        metrics = self.val_metrics(out, y_true)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def test_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Test step for a batch of graph
        """
        out = self.forward(batch)
        y_true = batch.y
        test_loss = self.loss_function(out, y_true)
        metrics = self.test_metrics(out, y_true)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"test_loss": test_loss}

    def predict_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """
        Prediction step returning raw model outputs.

        Args:
            batch: A Data object.
            batch_idx: Index of the batch.
        Returns:
            Inverse-transformed predictions if applicable.
        """
        if hasattr(self.model, "predict_step"):
            predictions = self.model.predict_step(batch, batch_idx)
        else:
            predictions = self.model(batch)
        return predictions
