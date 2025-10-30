"""Graph Lightning wrapper for PyTorch Geometric models.

This module provides a Lightning wrapper for graph neural networks that
work with PyTorch Geometric Data objects, integrating with the dlkit processing pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any

from torch import Tensor
from torch_geometric.data import Batch, Data

from dlkit.core.datatypes.networks import GraphDict, GraphInput
from dlkit.runtime.pipelines.classifiers import NameBasedClassifier
from dlkit.runtime.pipelines.graph_support import (
    GraphBatchPayload,
    GraphDataExtractionStep,
    GraphModelInvocationStep,
    PyGBatchAdapter,
)
from dlkit.runtime.pipelines.model_invokers import StandardModelInvoker
from dlkit.tools.config import ModelComponentSettings, WrapperComponentSettings
from dlkit.tools.config.data_entries import DataEntry

from .base import ProcessingLightningWrapper


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
        self._graph_adapter = kwargs.pop("graph_batch_adapter", None) or PyGBatchAdapter()

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
            LossPairingStep,
            OutputClassificationStep,
            OutputNamingStep,
            PrecisionValidationStep,
            ProcessingPipeline,
        )

        output_namer = self._create_output_namer()
        is_autoencoder = getattr(self._wrapper_settings, "is_autoencoder", False)

        def build_pipeline(include_loss_pairing: bool) -> ProcessingPipeline:
            local_invoker = GraphModelInvoker(self.model)
            if include_loss_pairing:
                next_step = LossPairingStep(entry_configs, is_autoencoder=is_autoencoder)
            else:
                next_step = None

            classification_step = OutputClassificationStep(
                output_classifier,
                OutputNamingStep(output_namer, next_step),
            )
            invocation_step = GraphModelInvocationStep(local_invoker, classification_step)
            precision_step = PrecisionValidationStep(local_invoker, invocation_step)
            extraction_step = GraphDataExtractionStep(entry_configs, self._graph_adapter, precision_step)
            return ProcessingPipeline(extraction_step)

        # Training pipeline with loss pairing
        self.train_pipeline = build_pipeline(include_loss_pairing=True)

        # Validation/test pipeline mirrors training pipeline
        self.val_pipeline = build_pipeline(include_loss_pairing=True)
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
            OutputClassificationStep,
            OutputNamingStep,
            PrecisionValidationStep,
            ProcessingPipeline,
        )

        precision_step = PrecisionValidationStep(
            model_invoker,
            GraphModelInvocationStep(
                model_invoker,
                OutputClassificationStep(
                    output_classifier,
                    OutputNamingStep(output_namer),
                ),
            ),
        )
        extraction_predict = GraphDataExtractionStep(entry_configs, self._graph_adapter, precision_step)
        self.predict_pipeline = ProcessingPipeline(extraction_predict)


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
        super().__init__(model, input_mode="kwargs")

    def invoke(self, features: GraphDict) -> GraphDict:  # type: ignore[override]
        """Invoke graph model using tensor keyword arguments."""
        return super().invoke(features)

    def invoke_with_payload(
        self,
        features: GraphDict,
        *,
        payload: GraphBatchPayload | None = None,
    ) -> GraphDict:
        """Invoke graph model with tensor kwargs and fallback to original batch."""
        try:
            return super().invoke(features)
        except RuntimeError as exc:
            if payload and isinstance(payload.original, (Data, Batch)):
                try:
                    outputs = self._model(payload.original)
                    return self._normalize_outputs(outputs)
                except Exception as fallback_exc:  # pragma: no cover - rare fallback path
                    raise RuntimeError(f"Graph model invocation failed: {fallback_exc}") from fallback_exc
            raise exc
