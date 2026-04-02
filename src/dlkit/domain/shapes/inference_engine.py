"""Shape inference engine — internal implementation.

This module contains the inference engine and context that coordinate
shape inference strategies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import IShapeSpec, create_shape_spec
from .inference_strategies import (
    CheckpointMetadataStrategy,
    ConfigurationStrategy,
    DatasetSamplingStrategy,
    DefaultFallbackStrategy,
    GraphDatasetStrategy,
    ShapeInferenceStrategy,
)
from .value_objects import ModelFamily


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceContext:
    """Context object containing all information needed for shape inference.

    This context object follows the Parameter Object pattern to avoid
    long parameter lists and provide extensibility for future inference sources.
    """

    dataset: Any = None
    checkpoint_path: Path | None = None
    model_settings: Any = None
    entry_configs: dict[str, Any] | None = None
    model_family: ModelFamily | None = None
    shape_factory: Any = None  # ShapeSystemFactory - avoid circular import


class ShapeInferenceChain:
    """Chain of responsibility for executing shape inference strategies.

    This chain tries multiple strategies in priority order until one succeeds,
    providing robust shape inference with graceful fallbacks.
    """

    def __init__(self, strategies: list[ShapeInferenceStrategy] | None = None):
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

    def get_strategies(self) -> list[ShapeInferenceStrategy]:
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

    def _create_default_strategies(self) -> list[ShapeInferenceStrategy]:
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
        inference_chain: ShapeInferenceChain | None = None,
        shape_factory: Any | None = None,
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
        entry_configs: dict[str, Any] | None = None,
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
        checkpoint_path: Path | None = None,
        dataset: Any = None,
        model_settings: Any = None,
        entry_configs: dict[str, Any] | None = None,
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
