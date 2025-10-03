from collections.abc import Callable
from inspect import Parameter, isclass, signature
from typing import Any, Self

import torch
from lightning import LightningModule
from torch_geometric.data import Data

from dlkit.tools.config import ModelComponentSettings
from dlkit.tools.config.core.base_settings import ComponentSettings
from dlkit.tools.utils.general import kwargs_compatible_with
from dlkit.tools.io.system import load_class
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
        settings: ModelComponentSettings,
    ) -> None:
        super().__init__()
        # Save hyperparameters for reproducibility
        super().save_hyperparameters(settings.to_dict())
        self.settings = settings

        # Initialize the graph neural network model
        self.model = _build_component(settings)

        # Loss function from model or settings
        self.loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = getattr(
            self.model, "loss_function", None
        ) or _build_component(self.settings.loss_function)

        # Metrics for validation and testing
        self.val_metrics = MetricCollection(
            {metric.name: _build_component(metric) for metric in self.settings.metrics}
        )
        self.test_metrics = MetricCollection(
            {metric.name: _build_component(metric) for metric in self.settings.metrics}
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

    def configure_optimizers(self) -> dict[str, Any] | torch.optim.Optimizer:
        """
        Configure optimizer and scheduler from settings.

        Returns:
            Dictionary or optimizer for Lightning.
        """
        optimizer = _build_component(
            self.settings.optimizer, params=self.model.parameters()
        )
        scheduler = _build_component(self.settings.scheduler, optimizer=optimizer)
        if scheduler is None:
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
        loss = self.loss_fn(out, y_true)
        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Validation step for a batch of graph
        """
        out = self.forward(batch)
        y_true = batch.y
        val_loss = self.loss_fn(out, y_true)
        metrics = self.val_metrics(out, y_true)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": val_loss}

    def on_validation_epoch_end(self) -> None:
        """
        Reset validation metrics at the end of the epoch.
        """
        self.val_metrics.reset()

    def test_step(self, batch: Data, batch_idx: int) -> dict[str, torch.Tensor]:
        """
        Test step for a batch of graph
        """
        out = self.forward(batch)
        y_true = batch.y
        test_loss = self.loss_fn(out, y_true)
        metrics = self.test_metrics(out, y_true)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
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
        if hasattr(self.model, "predict_step"):
            predictions = self.model.predict_step(batch, batch_idx)
        else:
            predictions = self.model(batch)
        return predictions

    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str, **kwargs) -> Self:
        """
        Load the PipelineGeometricNetwork from a Lightning checkpoint.
        """
        data = torch.load(ckpt_path)
        hparams = data.get("hyper_parameters") or data["hparams"]
        settings = ModelComponentSettings.model_validate(hparams)
        return super().load_from_checkpoint(ckpt_path, model_settings=settings, **kwargs)


def _build_component(
    component: ComponentSettings[Any] | Callable[..., Any] | Any,
    **overrides: Any,
) -> Any:
    """Instantiate a component described by settings or return existing callables.

    This helper mirrors the factory behavior used across the modern settings system,
    allowing legacy graph wrappers to continue operating with ComponentSettings
    while keeping type checking happy.
    """

    if component is None:
        return None

    if isinstance(component, ComponentSettings):
        init_kwargs = component.get_init_kwargs()
        init_kwargs.update(overrides)

        name = component.name
        module_path = component.module_path or ""

        if not isinstance(name, str):
            target = name
        else:
            if not module_path:
                msg = "Component settings must define a module_path for dynamic loading"
                raise ValueError(msg)
            target = load_class(name, module_path)

        if isclass(target):
            filtered_kwargs = kwargs_compatible_with(target, **init_kwargs)
            return target(**filtered_kwargs)

        if callable(target):
            target_signature = signature(target)
            accepts_kwargs = any(
                param.kind == Parameter.VAR_KEYWORD for param in target_signature.parameters.values()
            )
            if accepts_kwargs:
                return target(**init_kwargs)
            filtered_kwargs = {
                key: value for key, value in init_kwargs.items() if key in target_signature.parameters
            }
            return target(**filtered_kwargs)
        return target

    if callable(component):
        if isclass(component):
            filtered_kwargs = kwargs_compatible_with(component, **overrides)
            return component(**filtered_kwargs)
        component_signature = signature(component)
        accepts_kwargs = any(
            param.kind == Parameter.VAR_KEYWORD for param in component_signature.parameters.values()
        )
        if accepts_kwargs:
            return component(**overrides)
        filtered_overrides = {
            key: value for key, value in overrides.items() if key in component_signature.parameters
        }
        return component(**filtered_overrides)

    return component
