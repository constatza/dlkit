"""Enhanced shape inference strategy system.

This module provides a completely rewritten shape inference system using
the Strategy pattern and Chain of Responsibility pattern for flexible
and extensible shape inference from multiple sources.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from contextlib import suppress
from typing import Any, Dict, List, Optional, cast
from dataclasses import dataclass

from .value_objects import ShapeData, ShapeEntry, ModelFamily, ShapeSource
from .core import IShapeSpec, create_shape_spec
# Avoid circular import - factory imported at runtime where needed


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceContext:
    """Context object containing all information needed for shape inference.

    This context object follows the Parameter Object pattern to avoid
    long parameter lists and provide extensibility for future inference sources.
    """

    dataset: Any = None
    checkpoint_path: Optional[Path] = None
    model_settings: Any = None
    entry_configs: Optional[Dict[str, Any]] = None
    model_family: Optional[ModelFamily] = None
    shape_factory: Any = None  # ShapeSystemFactory - avoid circular import

    def __post_init__(self):
        """Initialize default factory if not provided."""
        if self.shape_factory is None:
            from .factory import ShapeSystemFactory

            object.__setattr__(self, "shape_factory", ShapeSystemFactory.create_production_system())


class ShapeInferenceStrategy(ABC):
    """Abstract base class for shape inference strategies.

    Each strategy implements a specific method of inferring shape information
    and can determine whether it's applicable to a given context.
    """

    @abstractmethod
    def can_infer(self, context: InferenceContext) -> bool:
        """Check if this strategy can infer shapes from the given context.

        Args:
            context: Inference context containing available data sources

        Returns:
            True if this strategy can attempt inference with the given context
        """
        ...

    @abstractmethod
    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Infer shape data from the context.

        Args:
            context: Inference context containing data sources

        Returns:
            ShapeData if successful, None if inference fails
        """
        ...

    @abstractmethod
    def get_priority(self) -> int:
        """Get the priority of this strategy for ordering.

        Lower numbers indicate higher priority.

        Returns:
            Priority value (0 = highest priority)
        """
        ...

    def get_name(self) -> str:
        """Get a human-readable name for this strategy.

        Returns:
            Strategy name for logging and debugging
        """
        return self.__class__.__name__


class CheckpointMetadataStrategy(ShapeInferenceStrategy):
    """Strategy to infer shapes from enhanced checkpoint metadata."""

    def can_infer(self, context: InferenceContext) -> bool:
        """Check if checkpoint path exists and is readable.

        Args:
            context: Inference context

        Returns:
            True if checkpoint path is available
        """
        return (
            context.checkpoint_path is not None
            and context.checkpoint_path.exists()
            and context.checkpoint_path.is_file()
        )

    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Extract shapes from enhanced checkpoint metadata.

        Args:
            context: Inference context containing checkpoint path

        Returns:
            ShapeData from checkpoint metadata or None if extraction fails
        """
        if context.checkpoint_path is None:
            return None
        with suppress(Exception):
            import torch

            checkpoint = torch.load(context.checkpoint_path, map_location="cpu", weights_only=False)
            if "dlkit_metadata" in checkpoint and "shape_spec" in checkpoint["dlkit_metadata"]:
                return context.shape_factory.get_serializer().deserialize(
                    checkpoint["dlkit_metadata"]["shape_spec"]
                )
        return None

    def get_priority(self) -> int:
        """Highest priority - enhanced metadata is most reliable."""
        return 0


class GraphDatasetStrategy(ShapeInferenceStrategy):
    """Strategy to infer shapes from PyTorch Geometric datasets."""

    @staticmethod
    def _pyg_types() -> tuple[type | None, type | None]:
        try:
            from torch_geometric.data import Data, Batch  # type: ignore
        except Exception:  # pragma: no cover - PyG optional at import time
            return (None, None)
        return (Data, Batch)

    def can_infer(self, context: InferenceContext) -> bool:
        """Graph inference requires a dataset with PyG Data samples."""
        if context.dataset is None:
            return False
        sample = self._safe_sample(context.dataset)
        return self._is_graph_sample(sample)

    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Infer graph-centric shapes using PyG metadata."""
        sample = self._safe_sample(context.dataset)
        graph = self._extract_graph_sample(sample)
        if graph is None:
            return None

        shapes = self._collect_graph_shapes(graph)
        if not shapes:
            return None

        entries = {name: ShapeEntry(name=name, dimensions=dims) for name, dims in shapes.items()}
        default_output = "y" if "y" in shapes else None
        return ShapeData(
            entries=entries,
            model_family=ModelFamily.GRAPH,
            source=ShapeSource.GRAPH_DATASET,
            default_input="x",
            default_output=default_output,
        )

    def get_priority(self) -> int:
        """High priority for graph datasets (after checkpoint metadata)."""
        return 1

    def _safe_sample(self, dataset: Any) -> Any:
        try:
            return dataset[0]
        except Exception:
            return None

    def _is_graph_sample(self, sample: Any) -> bool:
        Data, Batch = self._pyg_types()
        if Data is None or Batch is None:
            return False
        if isinstance(sample, (Data, Batch)):
            return True
        if isinstance(sample, dict):
            return any(self._is_graph_sample(item) for item in sample.values())
        if isinstance(sample, (list, tuple)):
            return any(self._is_graph_sample(item) for item in sample)
        return False

    def _extract_graph_sample(self, sample: Any):
        Data, Batch = self._pyg_types()
        if Data is None or Batch is None:
            return None
        if isinstance(sample, (Data, Batch)):
            return sample
        if isinstance(sample, dict):
            for item in sample.values():
                graph = self._extract_graph_sample(item)
                if graph is not None:
                    return graph
        if isinstance(sample, (list, tuple)):
            for item in sample:
                graph = self._extract_graph_sample(item)
                if graph is not None:
                    return graph
        return None

    def _collect_graph_shapes(self, graph: Any) -> Dict[str, tuple[int, ...]]:
        shapes: Dict[str, tuple[int, ...]] = {}

        x = getattr(graph, "x", None)
        if x is not None and hasattr(x, "shape") and x.shape:
            feature_dim = int(x.shape[-1]) if len(x.shape) > 1 else int(x.shape[0])
            if feature_dim > 0:
                shapes["x"] = (feature_dim,)

        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None and hasattr(edge_attr, "shape") and edge_attr.shape:
            attr_dim = int(edge_attr.shape[-1]) if len(edge_attr.shape) > 1 else 1
            if attr_dim > 0:
                shapes["edge_attr"] = (attr_dim,)

        edge_weight = getattr(graph, "edge_weight", None)
        if edge_weight is not None and hasattr(edge_weight, "shape") and edge_weight.shape:
            shapes["edge_weight"] = (1,)

        y = getattr(graph, "y", None)
        if y is not None and hasattr(y, "shape") and y.shape:
            if y.ndim == 0:
                shapes["y"] = (1,)
            elif y.ndim == 1:
                target_dim = int(y.shape[0])
                if target_dim > 0:
                    shapes["y"] = (target_dim,)
            else:
                target_dim = int(y.shape[-1])
                if target_dim > 0:
                    shapes["y"] = (target_dim,)

        return shapes


class DatasetSamplingStrategy(ShapeInferenceStrategy):
    """Strategy to infer shapes by sampling from dataset."""

    def can_infer(self, context: InferenceContext) -> bool:
        """Check if dataset is available for sampling.

        Args:
            context: Inference context

        Returns:
            True if dataset is available
        """
        return context.dataset is not None

    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Infer shapes by sampling dataset.

        Args:
            context: Inference context containing dataset

        Returns:
            ShapeData inferred from dataset sampling or None if sampling fails
        """
        try:
            shapes = self._sample_dataset_shapes(context.dataset)
            if not shapes:
                return None

            # Enhance with entry config information if available
            if context.entry_configs:
                shapes = self._enhance_with_entry_configs(shapes, context.entry_configs)

            # Determine model family
            model_family = ModelFamily.DLKIT_NN
            if context.model_family:
                model_family = context.model_family
            elif context.model_settings:
                model_family = context.shape_factory.get_model_registry().detect_family(
                    context.model_settings
                )

            # Create shape entries
            entries = {
                name: ShapeEntry(name=name, dimensions=dims) for name, dims in shapes.items()
            }

            return ShapeData(
                entries=entries, model_family=model_family, source=ShapeSource.TRAINING_DATASET
            )

        except Exception:
            return None

    def _sample_dataset_shapes(self, dataset: Any) -> Dict[str, tuple[int, ...]] | None:
        """Sample dataset to extract shape information."""
        from torch import Tensor

        try:
            sample = dataset[0]
        except Exception:
            return None

        shapes: Dict[str, tuple[int, ...]] = {}

        with suppress(ImportError):
            from tensordict import TensorDictBase

            if isinstance(sample, TensorDictBase) and "features" in sample and "targets" in sample:
                feat_td = cast(TensorDictBase, sample["features"])
                targ_td = cast(TensorDictBase, sample["targets"])
                for i, key in enumerate(feat_td.keys()):
                    tensor = feat_td[key]
                    if isinstance(tensor, Tensor):
                        shapes[f"x{i}" if i > 0 else "x"] = tuple(int(d) for d in tensor.shape)
                for i, key in enumerate(targ_td.keys()):
                    tensor = targ_td[key]
                    if isinstance(tensor, Tensor):
                        shapes[f"y{i}" if i > 0 else "y"] = tuple(int(d) for d in tensor.shape)
                return shapes if shapes else None

        match sample:
            case dict():
                for name, tensor in sample.items():
                    if isinstance(tensor, Tensor):
                        shapes[name] = tuple(int(d) for d in tensor.shape)
            case list() | tuple():
                if len(sample) >= 1 and isinstance(sample[0], Tensor):
                    shapes["x"] = tuple(int(d) for d in sample[0].shape)
                if len(sample) > 1 and isinstance(sample[1], Tensor):
                    shapes["y"] = tuple(int(d) for d in sample[1].shape)
            case _ if isinstance(sample, Tensor):
                shapes["x"] = tuple(int(d) for d in sample.shape)

        return shapes if shapes else None

    def _enhance_with_entry_configs(
        self, shapes: Dict[str, tuple[int, ...]], entry_configs: Dict[str, Any]
    ) -> Dict[str, tuple[int, ...]]:
        """Enhance shapes with information from entry configurations."""
        enhanced = dict(shapes)

        # Add x/y aliases based on entry configs
        features = {
            name: cfg
            for name, cfg in entry_configs.items()
            if hasattr(cfg, "__class__") and "Feature" in cfg.__class__.__name__
        }
        targets = {
            name: cfg
            for name, cfg in entry_configs.items()
            if hasattr(cfg, "__class__") and "Target" in cfg.__class__.__name__
        }

        if len(features) == 1 and "x" not in enhanced:
            feat_name = next(iter(features.keys()))
            if feat_name in enhanced:
                enhanced["x"] = enhanced[feat_name]

        if len(targets) == 1 and "y" not in enhanced:
            targ_name = next(iter(targets.keys()))
            if targ_name in enhanced:
                enhanced["y"] = enhanced[targ_name]

        return enhanced

    def get_priority(self) -> int:
        """Medium priority - reliable but requires dataset."""
        return 2


class ConfigurationStrategy(ShapeInferenceStrategy):
    """Strategy to infer shapes from explicit model configuration."""

    def can_infer(self, context: InferenceContext) -> bool:
        """Check if model settings contain shape information.

        Args:
            context: Inference context

        Returns:
            True if model settings might contain shape info
        """
        return context.model_settings is not None

    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Infer shapes from model configuration.

        Args:
            context: Inference context containing model settings

        Returns:
            ShapeData from configuration or None if not available
        """
        try:
            settings = context.model_settings

            # Look for explicit shape configuration
            shapes = {}
            if hasattr(settings, "input_shape") and settings.input_shape:
                shapes["x"] = tuple(settings.input_shape)
            if hasattr(settings, "output_shape") and settings.output_shape:
                shapes["y"] = tuple(settings.output_shape)

            # Look for shape dictionary
            if hasattr(settings, "shapes") and settings.shapes:
                shapes.update(settings.shapes)

            if not shapes:
                return None

            # Determine model family
            model_family = ModelFamily.DLKIT_NN
            if context.model_family:
                model_family = context.model_family
            else:
                model_family = context.shape_factory.get_model_registry().detect_family(settings)

            # Create shape entries
            entries = {
                name: ShapeEntry(name=name, dimensions=dims) for name, dims in shapes.items()
            }

            return ShapeData(
                entries=entries, model_family=model_family, source=ShapeSource.CONFIGURATION
            )

        except Exception:
            return None

    def get_priority(self) -> int:
        """Low priority - configuration might be incomplete."""
        return 3


class DefaultFallbackStrategy(ShapeInferenceStrategy):
    """Strategy providing sensible default shapes as final fallback."""

    def can_infer(self, context: InferenceContext) -> bool:
        """Always can provide default shapes.

        Args:
            context: Inference context

        Returns:
            Always True - can always provide defaults
        """
        return True

    def infer_shapes(self, context: InferenceContext) -> Optional[ShapeData]:
        """Provide default shapes as final fallback.

        Args:
            context: Inference context

        Returns:
            ShapeData with minimal default shapes
        """
        # Determine model family
        model_family = ModelFamily.EXTERNAL
        if context.model_family:
            model_family = context.model_family
        elif context.model_settings:
            model_family = context.shape_factory.get_model_registry().detect_family(
                context.model_settings
            )

        # Provide appropriate defaults based on model family
        if model_family == ModelFamily.EXTERNAL:
            # External models get empty shapes
            entries = {}
        else:
            # Other models get minimal x/y shapes
            entries = {
                "x": ShapeEntry(name="x", dimensions=(1,)),
                "y": ShapeEntry(name="y", dimensions=(1,)),
            }

        return ShapeData(
            entries=entries, model_family=model_family, source=ShapeSource.DEFAULT_FALLBACK
        )

    def get_priority(self) -> int:
        """Lowest priority - final fallback."""
        return 999


class ShapeInferenceChain:
    """Chain of responsibility for executing shape inference strategies.

    This chain tries multiple strategies in priority order until one succeeds,
    providing robust shape inference with graceful fallbacks.
    """

    def __init__(self, strategies: Optional[List[ShapeInferenceStrategy]] = None):
        """Initialize inference chain with strategies.

        Args:
            strategies: List of strategies in priority order (creates defaults if None)
        """
        if strategies is None:
            self._strategies = self._create_default_strategies()
        else:
            self._strategies = strategies

        # Sort by priority (lower numbers = higher priority)
        self._strategies.sort(key=lambda s: s.get_priority())

    def infer_shape_spec(self, context: InferenceContext) -> IShapeSpec:
        """Execute shape inference chain until successful.

        Args:
            context: Inference context containing data sources

        Returns:
            IShapeSpec from first successful strategy

        Raises:
            ValueError: If all strategies fail (should not happen with DefaultFallbackStrategy)
        """
        for strategy in self._strategies:
            if strategy.can_infer(context):
                try:
                    shape_data = strategy.infer_shapes(context)
                    if shape_data is not None:
                        # Create shape spec from inferred data
                        shapes = {
                            name: entry.dimensions for name, entry in shape_data.entries.items()
                        }
                        return create_shape_spec(
                            shapes=shapes,
                            model_family=shape_data.model_family,
                            source=shape_data.source,
                        )
                except Exception:
                    # Continue to next strategy if this one fails
                    continue

        # This should not happen if DefaultFallbackStrategy is included
        raise ValueError("All shape inference strategies failed")

    def get_strategies(self) -> List[ShapeInferenceStrategy]:
        """Get list of all strategies in priority order.

        Returns:
            List of inference strategies
        """
        return self._strategies.copy()

    def add_strategy(self, strategy: ShapeInferenceStrategy) -> None:
        """Add a new strategy to the chain.

        Args:
            strategy: Strategy to add
        """
        self._strategies.append(strategy)
        # Re-sort by priority
        self._strategies.sort(key=lambda s: s.get_priority())

    def _create_default_strategies(self) -> List[ShapeInferenceStrategy]:
        """Create default strategy list.

        Returns:
            List of default inference strategies
        """
        return [
            CheckpointMetadataStrategy(),
            GraphDatasetStrategy(),
            DatasetSamplingStrategy(),
            ConfigurationStrategy(),
            DefaultFallbackStrategy(),
        ]


class ShapeInferenceEngine:
    """High-level interface for shape inference operations.

    This engine provides a simple interface for shape inference while
    hiding the complexity of the strategy chain underneath.
    """

    def __init__(
        self,
        inference_chain: Optional[ShapeInferenceChain] = None,
        shape_factory: Optional[Any] = None,
    ):  # ShapeSystemFactory - avoid circular import
        """Initialize inference engine.

        Args:
            inference_chain: Optional inference chain (creates default if None)
            shape_factory: Optional shape factory (creates default if None)
        """
        self._inference_chain = inference_chain or ShapeInferenceChain()
        if shape_factory is None:
            from .factory import ShapeSystemFactory

            self._shape_factory = ShapeSystemFactory.create_production_system()
        else:
            self._shape_factory = shape_factory

    def infer_from_checkpoint(
        self, checkpoint_path: Path, model_settings: Any = None
    ) -> IShapeSpec:
        """Infer shapes from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
            model_settings: Optional model settings for family detection

        Returns:
            IShapeSpec with inferred shapes
        """
        context = InferenceContext(
            checkpoint_path=checkpoint_path,
            model_settings=model_settings,
            shape_factory=self._shape_factory,
        )
        return self._inference_chain.infer_shape_spec(context)

    def infer_from_dataset(
        self,
        dataset: Any,
        model_settings: Any = None,
        entry_configs: Optional[Dict[str, Any]] = None,
    ) -> IShapeSpec:
        """Infer shapes from dataset.

        Args:
            dataset: Dataset to sample from
            model_settings: Optional model settings for family detection
            entry_configs: Optional entry configurations for enhanced inference

        Returns:
            IShapeSpec with inferred shapes
        """
        context = InferenceContext(
            dataset=dataset,
            model_settings=model_settings,
            entry_configs=entry_configs,
            shape_factory=self._shape_factory,
        )
        return self._inference_chain.infer_shape_spec(context)

    def infer_from_config(self, model_settings: Any) -> IShapeSpec:
        """Infer shapes from model configuration.

        Args:
            model_settings: Model settings containing shape information

        Returns:
            IShapeSpec with inferred shapes
        """
        context = InferenceContext(model_settings=model_settings, shape_factory=self._shape_factory)
        return self._inference_chain.infer_shape_spec(context)

    def infer_comprehensive(
        self,
        checkpoint_path: Optional[Path] = None,
        dataset: Any = None,
        model_settings: Any = None,
        entry_configs: Optional[Dict[str, Any]] = None,
    ) -> IShapeSpec:
        """Comprehensive shape inference using all available sources.

        Args:
            checkpoint_path: Optional checkpoint file path
            dataset: Optional dataset to sample from
            model_settings: Optional model settings
            entry_configs: Optional entry configurations

        Returns:
            IShapeSpec with inferred shapes using best available source
        """
        context = InferenceContext(
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            model_settings=model_settings,
            entry_configs=entry_configs,
            shape_factory=self._shape_factory,
        )
        return self._inference_chain.infer_shape_spec(context)
