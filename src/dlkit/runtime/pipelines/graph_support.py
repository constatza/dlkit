"""Graph-specific pipeline support utilities.

This module provides graph-aware pipeline components that preserve the
original ``torch_geometric`` batch objects while still integrating with the
standard dlkit processing pipeline. The goal is to avoid expensive batch
reconstruction while keeping model invocation compatible with tensor-based
forwards required for checkpointing and export workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from loguru import logger
from torch import Tensor
from torch_geometric.data import Batch, Data

from dlkit.core.datatypes.networks import GraphInput
from dlkit.tools.config.data_entries import DataEntry

from .context import ProcessingContext
from .pipeline import DataExtractionStep, ModelInvocationStep

GRAPH_PAYLOAD_KEY = "graph_payload"


@dataclass(frozen=True)
class GraphBatchPayload:
    """Container preserving both the original batch and tensor mapping."""

    original: GraphInput
    tensor_map: dict[str, Tensor]


class GraphBatchAdapter(Protocol):
    """Adapter interface for transforming raw graph batches."""

    def prepare(self, batch: GraphInput) -> GraphBatchPayload:
        """Normalize raw graph input into a payload with tensor mapping."""


class PyGBatchAdapter:
    """Default adapter for torch-geometric ``Data``/``Batch`` instances."""

    _COMMON_ATTRS = ("x", "edge_index", "edge_attr", "y", "batch")

    def prepare(self, batch: GraphInput) -> GraphBatchPayload:  # type: ignore[override]
        if isinstance(batch, dict):
            tensor_map = {key: value for key, value in batch.items() if isinstance(value, Tensor)}
            return GraphBatchPayload(original=batch, tensor_map=tensor_map)

        if isinstance(batch, (Data, Batch)):
            tensor_map: dict[str, Tensor] = {}
            for key in batch.keys():
                value = getattr(batch, key)
                if isinstance(value, Tensor):
                    tensor_map[key] = value

            for attr in self._COMMON_ATTRS:
                if hasattr(batch, attr):
                    value = getattr(batch, attr)
                    if isinstance(value, Tensor):
                        tensor_map.setdefault(attr, value)

            return GraphBatchPayload(original=batch, tensor_map=tensor_map)

        # Fallback for graph-like custom objects
        if hasattr(batch, "x"):
            tensor_map = {}
            for attr in self._COMMON_ATTRS:
                if hasattr(batch, attr):
                    value = getattr(batch, attr)
                    if isinstance(value, Tensor):
                        tensor_map[attr] = value

            if tensor_map:
                return GraphBatchPayload(original=batch, tensor_map=tensor_map)

        raise TypeError(f"Unsupported graph batch type: {type(batch)}")


class GraphDataExtractionStep(DataExtractionStep):
    """Data extraction step that preserves the original PyG batch."""

    def __init__(
        self,
        entry_configs: dict[str, DataEntry],
        batch_adapter: GraphBatchAdapter,
        next_step=None,
        *,
        payload_key: str = GRAPH_PAYLOAD_KEY,
    ):
        super().__init__(entry_configs, next_step)
        self._batch_adapter = batch_adapter
        self._payload_key = payload_key

    def process(self, context: ProcessingContext) -> ProcessingContext:
        payload = self._batch_adapter.prepare(context.raw_batch)
        context.artifacts[self._payload_key] = payload
        context.raw_batch = payload.tensor_map
        return super().process(context)


class GraphModelInvocationStep(ModelInvocationStep):
    """Model invocation that reuses the preserved PyG batch when available."""

    def __init__(
        self,
        model_invoker,
        next_step=None,
        *,
        payload_key: str = GRAPH_PAYLOAD_KEY,
    ):
        super().__init__(model_invoker, next_step)
        self._payload_key = payload_key

    def process(self, context: ProcessingContext) -> ProcessingContext:  # type: ignore[override]
        if not context.features:
            raise RuntimeError("No features available for model invocation")

        model = getattr(self._model_invoker, "model", None)
        if hasattr(model, "dtype"):
            model_dtype = model.dtype
            for name, tensor in context.features.items():
                if tensor.is_floating_point() and tensor.dtype != model_dtype:
                    logger.debug(
                        f"Feature '{name}' dtype {tensor.dtype} differs from current model dtype "
                        f"{model_dtype}. Lightning's precision plugin will handle alignment during "
                        "forward pass."
                    )

        payload = context.artifacts.get(self._payload_key)

        try:
            if hasattr(self._model_invoker, "invoke_with_payload"):
                outputs = self._model_invoker.invoke_with_payload(  # type: ignore[attr-defined]
                    context.features,
                    payload=payload,
                )
            else:
                outputs = self._model_invoker.invoke(context.features)
            context.model_outputs = outputs
        except Exception as exc:
            raise RuntimeError(f"Model invocation step failed: {exc}") from exc

        return self._next_step.handle(context) if self._next_step else context
