"""Graph Lightning wrapper for PyTorch Geometric models.

This module provides a Lightning wrapper for graph neural networks that
work with PyTorch Geometric Data objects, integrating with the dlkit processing pipeline.
"""

from typing import Any

from torch import Tensor
from torch_geometric.data import Data

from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry, Target
from dlkit.runtime.pipelines.classifiers import NameBasedClassifier
from dlkit.runtime.pipelines.model_invokers import StandardModelInvoker
from dlkit.runtime.pipelines.context import ProcessingContext
from .base import ProcessingLightningWrapper
import warnings


class GraphLightningWrapper(ProcessingLightningWrapper):
    """Lightning wrapper for PyTorch Geometric graph neural networks.

    This wrapper handles models that accept PyG Data objects and provides
    graph-specific training, validation, and test steps using the processing pipeline.

    It's designed to be a drop-in replacement for the original GraphNetwork
    but with enhanced processing capabilities.

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
            entry_configs: Data entry configurations for pipeline setup
            **kwargs: Additional arguments passed to base class
        """
        super().__init__(
            settings=settings,
            model_settings=model_settings,
            entry_configs=entry_configs,
            **kwargs,
        )

    def forward(self, data: Data) -> Tensor:
        """Forward pass through the model with PyG Data input.

        Args:
            data: PyTorch Geometric Data object containing x, edge_index, etc.

        Returns:
            Model output tensor
        """
        return self.model(data)

    def _create_output_classifier(self):
        """Create name-based output classifier for graph models.

        Graph models often have multiple named outputs, so we use
        name-based classification as the default strategy.

        Returns:
            NameBasedClassifier instance
        """
        return NameBasedClassifier()

    def _setup_processing_pipelines(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up processing pipelines optimized for graph

        Args:
            entry_configs: Data entry configurations
        """
        # Create graph-specific model invoker
        model_invoker = GraphModelInvoker(self.model)

        # Create output classifier
        output_classifier = self._create_output_classifier()
        # Warn and skip dlkit transforms for PyG graphs to avoid conflicts with PyG transforms
        try:
            has_transforms = any(getattr(cfg, "transforms", None) for cfg in entry_configs.values())
            if has_transforms:
                warnings.warn(
                    "dlkit transforms are ignored for graph datasets; use PyG pre_transform/transform instead.",
                    UserWarning,
                )
        except Exception:
            pass

        # Set up pipelines with graph-specific components (no dlkit TransformApplicationStep)
        from dlkit.runtime.pipelines.pipeline import (
            ProcessingPipeline,
            DataExtractionStep,
            ModelInvocationStep,
            OutputClassificationStep,
            OutputNamingStep,
            LossDataAggregationStep,
            ValidationDataStep,
        )

        # Training pipeline
        # Create namer (use base default)
        output_namer = self._create_output_namer()

        extraction_train = DataExtractionStep(entry_configs)
        invocation_train = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(
                    output_namer,
                    LossDataAggregationStep(entry_configs),
                ),
            ),
        )
        extraction_train.set_next(invocation_train)
        self.train_pipeline = ProcessingPipeline(extraction_train)

        # Validation and test pipelines
        extraction_val = DataExtractionStep(entry_configs)
        invocation_val = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer, ValidationDataStep()),
            ),
        )
        extraction_val.set_next(invocation_val)
        self.val_pipeline = ProcessingPipeline(extraction_val)

        self.test_pipeline = self.val_pipeline

    def _setup_predict_pipeline(self, entry_configs: dict[str, DataEntry]) -> None:
        """Set up inference-only processing pipeline for graph models.

        Args:
            entry_configs (dict[str, DataEntry]): Mapping from entry name to configuration.
        """
        # Create graph-specific model invoker
        model_invoker = GraphModelInvoker(self.model)

        # Create output classifier and namer
        output_classifier = self._create_output_classifier()
        output_namer = self._create_output_namer()

        # Build predict pipeline - direct: extraction → invocation (no transforms)
        from dlkit.runtime.pipelines.pipeline import (
            ProcessingPipeline,
            DataExtractionStep,
            ModelInvocationStep,
            OutputClassificationStep,
            OutputNamingStep,
        )

        extraction_predict = DataExtractionStep(entry_configs)
        invocation_predict = ModelInvocationStep(
            model_invoker,
            OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer)  # Terminates here - no loss pairing
            ),
        )
        extraction_predict.set_next(invocation_predict)
        self.predict_pipeline = ProcessingPipeline(extraction_predict)

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Training step for graph dataflow using processing pipeline.

        Args:
            batch: Dictionary containing graph data (or PyG Data object)
            batch_idx: Index of the batch

        Returns:
            Dictionary containing the training loss
        """
        # Handle both dict and PyG Data input formats
        if hasattr(batch, 'x'):  # PyG Data object
            batch_dict = self._data_to_dict(batch)  # type: ignore[arg-type]
        else:
            batch_dict = batch

        # Process through pipeline
        context = self.train_pipeline.execute(batch_dict)

        # Compute loss
        loss = self._compute_loss(context)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Validation step for graph dataflow using processing pipeline.

        Args:
            batch: Dictionary containing graph data (or PyG Data object)
            batch_idx: Index of the batch

        Returns:
            Dictionary containing validation metrics
        """
        # Handle both dict and PyG Data input formats
        if hasattr(batch, 'x'):  # PyG Data object
            batch_dict = self._data_to_dict(batch)  # type: ignore[arg-type]
        else:
            batch_dict = batch

        # Process through pipeline
        context = self.val_pipeline.execute(batch_dict)

        # Compute loss and metrics
        val_loss = self._compute_loss(context)
        metrics = self._compute_metrics(context, self.val_metrics)

        # Log metrics
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return {"val_loss": val_loss}

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Any]:
        """Test step for graph dataflow using processing pipeline.

        Args:
            batch: Dictionary containing graph data (or PyG Data object)
            batch_idx: Index of the batch

        Returns:
            Dictionary containing test metrics
        """
        # Handle both dict and PyG Data input formats
        if hasattr(batch, 'x'):  # PyG Data object
            batch_dict = self._data_to_dict(batch)  # type: ignore[arg-type]
        else:
            batch_dict = batch

        # Process through pipeline
        context = self.test_pipeline.execute(batch_dict)

        # Compute loss and metrics
        test_loss = self._compute_loss(context)
        metrics = self._compute_metrics(context, self.test_metrics)

        # Log metrics
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        return {"test_loss": test_loss}

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int):
        """Prediction step for graph dataflow using pipeline.

        Args:
            batch: Dictionary containing graph data (or PyG Data object)
            batch_idx: Index of the batch

        Returns:
            dict[str, torch.Tensor | dict]: Dictionary with ``predictions``, ``targets``, and ``latents``.
        """
        # Handle both dict and PyG Data input formats
        if hasattr(batch, 'x'):  # PyG Data object
            batch_dict = self._data_to_dict(batch)  # type: ignore[arg-type]
        else:
            batch_dict = batch
        context = self.predict_pipeline.execute(batch_dict)

        # No transforms for graph models - return predictions directly
        return {"predictions": dict(context.predictions), "targets": dict(context.targets), "latents": context.latents}

    def _data_to_dict(self, data: Data) -> dict[str, Tensor]:
        """Convert a PyG Data object to a dict of tensors.

        Args:
            data (torch_geometric.data.Data): PyG Data object.

        Returns:
            dict[str, torch.Tensor]: Dictionary representation of the
        """
        result = {}

        # Common PyG attributes
        if hasattr(data, "x") and data.x is not None:
            result["x"] = data.x
        if hasattr(data, "edge_index") and data.edge_index is not None:
            result["edge_index"] = data.edge_index
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            result["edge_attr"] = data.edge_attr
        if hasattr(data, "y") and data.y is not None:
            result["y"] = data.y
        if hasattr(data, "batch") and data.batch is not None:
            result["batch"] = data.batch

        # Additional attributes
        for key in data.keys():
            if key not in result and hasattr(data, key):
                value = getattr(data, key)
                if isinstance(value, Tensor):
                    result[key] = value

        return result



class GraphModelInvoker(StandardModelInvoker):
    """Model invoker specialized for graph neural networks.

    This invoker handles PyG Data objects and converts them to the
    appropriate format for model invocation.
    """

    def __init__(self, model):
        """Initialize the graph model invoker.

        Args:
            model: Graph neural network model
        """
        super().__init__(model, input_mode="single")

    def invoke(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """Invoke graph model with features.

        Args:
            features: Dictionary of feature tensors

        Returns:
            Dictionary of model outputs
        """
        # Convert features back to PyG Data format if needed
        data = self._dict_to_data(features)

        # Invoke model
        outputs = self._model(data)

        return self._normalize_outputs(outputs)

    def _dict_to_data(self, features: dict[str, Tensor]) -> Data:
        """Convert feature dictionary back to PyG Data object.

        Args:
            features: Dictionary of feature tensors

        Returns:
            PyG Data object
        """
        data = Data()

        for key, tensor in features.items():
            setattr(data, key, tensor)

        return data

