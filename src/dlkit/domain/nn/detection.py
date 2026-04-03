"""ABC-based model detection for SOLID compliance.

This module provides a clean, extensible model detection system that
replaces the hardcoded isinstance checks with ABC-based classification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from typing import Any

import torch.nn as nn


class ModelType(Enum):
    """Model type classifications."""

    SHAPE_AWARE_DLKIT = "shape_aware_dlkit"
    SHAPE_AGNOSTIC_EXTERNAL = "shape_agnostic_external"
    GRAPH = "graph"
    TIMESERIES = "timeseries"


class IModelTypeDetector(ABC):
    """Interface for model type detection strategies."""

    @abstractmethod
    def can_detect(self, model_settings: Any) -> bool:
        """Check if this detector can classify the model.

        Args:
            model_settings: Model configuration settings

        Returns:
            True if this detector can classify the model
        """
        ...

    @abstractmethod
    def detect_type(self, model_settings: Any) -> ModelType:
        """Detect the model type.

        Args:
            model_settings: Model configuration settings

        Returns:
            Detected model type
        """
        ...


class ABCModelTypeDetector(IModelTypeDetector):
    """ABC-based model type detector using inheritance checks."""

    def can_detect(self, model_settings: Any) -> bool:
        """Always can attempt detection."""
        return True

    def detect_type(self, model_settings: Any) -> ModelType:
        """Detect model type using class inheritance.

        Args:
            model_settings: Model configuration settings

        Returns:
            Model type based on class inheritance
        """
        model_cls = self._get_model_class(model_settings)

        if model_cls is None or not isinstance(model_cls, type):
            return ModelType.SHAPE_AGNOSTIC_EXTERNAL

        # Check inheritance
        try:
            from lightning.pytorch import LightningModule

            from dlkit.domain.nn.graph.base import BaseGraphNetwork

            if issubclass(model_cls, BaseGraphNetwork):
                return ModelType.GRAPH

            if issubclass(model_cls, LightningModule):
                return ModelType.SHAPE_AGNOSTIC_EXTERNAL

            if issubclass(model_cls, nn.Module):
                return ModelType.SHAPE_AWARE_DLKIT

        except ImportError:
            pass

        return ModelType.SHAPE_AGNOSTIC_EXTERNAL

    def _get_model_class(self, model_settings: Any) -> type[object] | Callable[..., object] | None:
        """Get the model class from settings.

        Args:
            model_settings: Model configuration settings

        Returns:
            Model class if available, None otherwise
        """
        try:
            model_name = getattr(model_settings, "name", None)
            if model_name is None:
                return None

            if isinstance(model_name, type):
                return model_name

            if isinstance(model_name, str):
                try:
                    return _import_object(
                        model_name, fallback_module=getattr(model_settings, "module_path", "")
                    )
                except Exception:
                    return None

        except Exception:
            pass

        return None


def _import_object(module_path: str, fallback_module: str = "") -> Callable[..., object]:
    """Import an object by ``module:attr`` or ``attr`` with a fallback module."""
    module_name, object_name = _split_module_path(module_path)
    resolved_module = module_name if module_name is not None else fallback_module
    module = import_module(resolved_module)
    obj = getattr(module, object_name)
    if obj is None:
        raise ImportError(f"Could not find {object_name} in {module_path}")
    return obj


def _split_module_path(path: str) -> tuple[str | None, str]:
    """Split ``module:attr`` notation into module and object parts."""
    if ":" not in path:
        return None, path
    module_name, object_name = path.split(":", 1)
    return module_name, object_name


class ModelTypeDetectionChain:
    """Chain of responsibility for model type detection."""

    def __init__(self, detectors: list[IModelTypeDetector] | None = None):
        """Initialize detection chain.

        Args:
            detectors: List of detectors in order of preference
        """
        self._detectors = detectors or [
            ABCModelTypeDetector(),
        ]

    def detect_model_type(self, model_settings: Any) -> ModelType:
        """Detect model type using chain of detectors.

        Args:
            model_settings: Model configuration settings

        Returns:
            Detected model type
        """
        for detector in self._detectors:
            if detector.can_detect(model_settings):
                return detector.detect_type(model_settings)

        # Should never reach here with ABCModelTypeDetector (always returns True for can_detect)
        return ModelType.SHAPE_AGNOSTIC_EXTERNAL


# Global instance for consistent detection
_detection_chain = ModelTypeDetectionChain()


def detect_model_type(model_settings: Any) -> ModelType:
    """Detect model type using the global detection chain.

    Args:
        model_settings: Model configuration settings

    Returns:
        Detected model type
    """
    return _detection_chain.detect_model_type(model_settings)


def requires_shape_spec(model_type: ModelType) -> bool:
    """Check if model type requires shape specification.

    Args:
        model_type: Model type to check

    Returns:
        True if model requires shape specification
    """
    return model_type in {
        ModelType.SHAPE_AWARE_DLKIT,
        ModelType.GRAPH,
        ModelType.TIMESERIES,
    }
