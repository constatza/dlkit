"""Domain models and value objects for inference.

Contains immutable domain objects that represent core business concepts
without external dependencies (pure domain objects).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from dlkit.core.shape_specs import ShapeSpec
from dlkit.tools.config.components.model_components import ModelComponentSettings

# Import only for type checking to avoid circular imports
if TYPE_CHECKING:
    from dlkit.core.training.transforms.chain import TransformChain


class ModelStateType(Enum):
    """Enumeration of model state types."""
    LOADED = "loaded"
    INFERENCE_READY = "inference_ready"
    ERROR = "error"


@dataclass(frozen=True)
class ModelState:
    """Immutable value object representing model state.

    This encapsulates the model and its associated metadata,
    ensuring consistent state management throughout the inference pipeline.

    Transforms are separated by data entry type (Feature vs Target) to maintain
    clear separation of concerns and eliminate runtime filtering.
    """
    model: Any  # PyTorch model
    state_type: ModelStateType
    device: str
    shape_spec: Optional[ShapeSpec] = None
    feature_transforms: Optional[Dict[str, TransformChain]] = None
    target_transforms: Optional[Dict[str, TransformChain]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None

    def is_inference_ready(self) -> bool:
        """Check if model is ready for inference execution."""
        return (
            self.state_type == ModelStateType.INFERENCE_READY
            and self.model is not None
            and self.error_info is None
        )

    def with_device(self, device: str) -> ModelState:
        """Create new ModelState with different device."""
        return ModelState(
            model=self.model,
            state_type=self.state_type,
            device=device,
            shape_spec=self.shape_spec,
            feature_transforms=self.feature_transforms,
            target_transforms=self.target_transforms,
            metadata=self.metadata,
            error_info=self.error_info
        )

    def with_inference_ready(self) -> ModelState:
        """Create new ModelState marked as inference ready."""
        return ModelState(
            model=self.model,
            state_type=ModelStateType.INFERENCE_READY,
            device=self.device,
            shape_spec=self.shape_spec,
            feature_transforms=self.feature_transforms,
            target_transforms=self.target_transforms,
            metadata=self.metadata,
            error_info=self.error_info
        )

    def with_error(self, error_info: Dict[str, Any]) -> ModelState:
        """Create new ModelState with error information."""
        return ModelState(
            model=self.model,
            state_type=ModelStateType.ERROR,
            device=self.device,
            shape_spec=self.shape_spec,
            feature_transforms=self.feature_transforms,
            target_transforms=self.target_transforms,
            metadata=self.metadata,
            error_info=error_info
        )


@dataclass(frozen=True)
class InferenceRequest:
    """Immutable value object representing an inference request.

    Contains all information needed to execute inference,
    separated from the technical concerns of model loading.
    """
    inputs: Dict[str, Any]
    batch_size: int
    apply_transforms: bool
    device: str = "auto"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_device(self, device: str) -> InferenceRequest:
        """Create new request with specified device."""
        return InferenceRequest(
            inputs=self.inputs,
            batch_size=self.batch_size,
            apply_transforms=self.apply_transforms,
            device=device,
            metadata=self.metadata
        )


@dataclass(frozen=True)
class InferenceContext:
    """Immutable context object for inference operations.

    Provides all available information for inference strategies
    without coupling them to specific data sources.
    """
    checkpoint_path: Path
    model_settings: Optional[ModelComponentSettings] = None
    dataset: Optional[Any] = None
    shape_spec: Optional[ShapeSpec] = None
    checkpoint_metadata: Dict[str, Any] = field(default_factory=dict)

    def has_enhanced_metadata(self) -> bool:
        """Check if context has enhanced checkpoint metadata."""
        return (
            "dlkit_metadata" in self.checkpoint_metadata
            and bool(self.checkpoint_metadata["dlkit_metadata"])
        )

    def has_dataset(self) -> bool:
        """Check if context has dataset for shape inference."""
        return self.dataset is not None

    def with_model_settings(
        self,
        model_settings: ModelComponentSettings
    ) -> InferenceContext:
        """Create new context with model settings."""
        return InferenceContext(
            checkpoint_path=self.checkpoint_path,
            model_settings=model_settings,
            dataset=self.dataset,
            shape_spec=self.shape_spec,
            checkpoint_metadata=self.checkpoint_metadata
        )

    def with_shape_spec(self, shape_spec: ShapeSpec) -> InferenceContext:
        """Create new context with shape specification."""
        return InferenceContext(
            checkpoint_path=self.checkpoint_path,
            model_settings=self.model_settings,
            dataset=self.dataset,
            shape_spec=shape_spec,
            checkpoint_metadata=self.checkpoint_metadata
        )

    def with_checkpoint_metadata(self, metadata: Dict[str, Any]) -> InferenceContext:
        """Create new context with checkpoint metadata."""
        return InferenceContext(
            checkpoint_path=self.checkpoint_path,
            model_settings=self.model_settings,
            dataset=self.dataset,
            shape_spec=self.shape_spec,
            checkpoint_metadata=metadata
        )


@dataclass(frozen=True)
class BatchProcessingConfig:
    """Configuration for batch processing during inference."""
    batch_size: int
    drop_last: bool = False
    num_workers: int = 0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")
