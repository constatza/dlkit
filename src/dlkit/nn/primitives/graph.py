from collections.abc import Callable
from typing import Any, Self

import torch
from lightning import LightningModule
from torch_geometric.data import Data

from dlkit.settings import ModelSettings
from dlkit.utils.loading import init_class
from torchmetrics import MetricCollection


class GraphNetwork(LightningModule):
    """
    A LightningModule wrapper for PyTorch Geometric networks.

    This class handles the training, validation, testing, and prediction
    loops for graph-based models built with PyTorch Geometric.
    Pure functions and model components are separated for clarity and testability.
    """

    def __init__(
        self,
        settings: ModelSettings,
    ) -> None:
        super().__init__()
        # Save hyperparameters for reproducibility
        super().save_hyperparameters(settings.model_dump(exclude_none=True))
        self.settings = settings

        # Initialize the graph neural network model
        self.model = init_class(settings)

        # Loss function from model or settings
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(
            self.model, "loss_function", None
        ) or init_class(self.settings.loss_function)

        # Metrics for validation and testing
        self.val_metrics = MetricCollection({m.name: init_class(m) for m in self.settings.metrics})
        self.test_metrics = MetricCollection({m.name: init_class(m) for m in self.settings.metrics})

    def forward(self, data: Data) -> torch.Tensor:
        """
        Perform a forward pass on a PyG Data object.

        Args:
            data: A torch_geometric.data.Data instance containing x, edge_index, etc.
        Returns:
            Model predictions (raw tensor).
        """
        return self.model(data)

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """
        Configure optimizer and scheduler from settings.

        Returns:
            Dictionary or optimizer for Lightning.
        """
        optimizer = init_class(self.settings.optimizer, params=self.model.parameters())
        scheduler = init_class(self.settings.scheduler, optimizer=optimizer)
        if not scheduler:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Training step for a batch of graph data.

        Args:
            batch: A Data object containing features, edge_index, and targets.
            batch_idx: Index of the batch.
        Returns:
            A dict containing the training loss.
        """
        # Forward pass
        out = self.forward(batch)
        # Extract ground truth
        y_true = batch.y
        # Compute loss
        loss = self.loss_fn(out, y_true)
        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Validation step for a batch of graph data.
        """
        out = self.forward(batch)
        y_true = batch.y
        val_loss = self.loss_fn(out, y_true)
        metrics = self.val_metrics(out, y_true)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self) -> None:
        """
        Reset validation metrics at the end of the epoch.
        """
        self.val_metrics.reset()

    def test_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Test step for a batch of graph data.
        """
        out = self.forward(batch)
        y_true = batch.y
        test_loss = self.loss_fn(out, y_true)
        metrics = self.test_metrics(out, y_true)
        self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_epoch=True)
        return {"test_loss": test_loss}

    def on_test_epoch_end(self) -> None:
        """
        Reset test metrics at the end of the epoch.
        """
        self.test_metrics.reset()

    def predict_step(self, batch: Data, batch_idx: int) -> torch.Tensor:
        """
        Prediction step returning raw model outputs.

        Args:
            batch: A Data object.
            batch_idx: Index of the batch.
        Returns:
            Inverse-transformed predictions if applicable.
        """
        preds = self.forward(batch)
        return preds

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, **kwargs) -> Self:
        """
        Load the PipelineGeometricNetwork from a Lightning checkpoint.
        """
        data = torch.load(ckpt_path)
        hparams = data.get("hyper_parameters") or data["hparams"]
        settings = ModelSettings.model_validate(hparams)
        return super().load_from_checkpoint(ckpt_path, model_settings=settings, **kwargs)
