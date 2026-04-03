"""Registry-based model family detection system.

This module implements the Registry pattern to replace hardcoded model family
detection with an extensible system that follows the Open/Closed principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn
from lightning.pytorch import LightningModule


def _string_attribute(settings: Any, name: str) -> str | None:
    """Safely extract a string attribute from *settings*.

    Args:
        settings: Settings object to extract from
        name: Attribute name to extract

    Returns:
        String value of the attribute or None if not found/convertible
    """
    value = getattr(settings, name, None)

    # Early return for None
    if value is None:
        return None

    # Already a string
    if isinstance(value, str):
        return value

    # Path-like object
    if hasattr(value, "__fspath__"):
        return str(value)

    # Bytes
    if isinstance(value, (bytes, bytearray)):
        return value.decode()

    # Enum with string value
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value

    # Object with string name
    if hasattr(value, "name") and isinstance(value.name, str):
        return value.name

    # Numbers
    if isinstance(value, (int, float)):
        return str(value)

    # Last resort: try str()
    try:
        return str(value)
    except Exception:
        return None


def _has_real_value(settings: Any, name: str) -> bool:
    """True when attribute *name* exists and is not None.

    Args:
        settings: Settings object to check
        name: Attribute name to check

    Returns:
        True if attribute exists and has a non-None value
    """
    if not hasattr(settings, name):
        return False
    value = getattr(settings, name)
    return value is not None


from .value_objects import ModelFamily


class ModelFamilyDetector(ABC):
    """Abstract base class for model family detection strategies.

    Each detector is responsible for determining if given model settings
    belong to a specific model family.
    """

    @abstractmethod
    def can_handle(self, settings: Any) -> bool:
        """Check if this detector can handle the given model settings.

        Args:
            settings: Model configuration settings

        Returns:
            True if this detector can identify the model family for these settings
        """
        ...

    @abstractmethod
    def get_priority(self) -> int:
        """Get the priority of this detector for ordering.

        Lower numbers indicate higher priority.

        Returns:
            Priority value (0 = highest priority)
        """
        ...

    def get_family(self) -> ModelFamily:
        """Get the model family this detector identifies.

        Default implementation extracts from class name.
        Override if different logic needed.

        Returns:
            ModelFamily enum value
        """
        class_name = self.__class__.__name__

        # Early return if doesn't end with 'Detector'
        if not class_name.endswith("Detector"):
            return ModelFamily.EXTERNAL

        # Extract family name and try to convert to enum
        family_name = class_name[:-8].lower()  # Remove 'Detector' suffix
        try:
            return ModelFamily(family_name)
        except ValueError:
            return ModelFamily.EXTERNAL


class ClassBasedDetector(ModelFamilyDetector):
    """Simple detector that imports the model class and checks its MODEL_FAMILY attribute.

    This replaces complex string pattern matching with direct class inspection,
    following the principle of using Python's type system instead of fragile string checks.
    """

    def __init__(self):
        """Initialize detector with storage for detected family."""
        self._detected_family: ModelFamily = ModelFamily.EXTERNAL

    def can_handle(self, settings: Any) -> bool:
        """Detect and store the model family, always returns True.

        Args:
            settings: Model configuration settings

        Returns:
            Always True (detection happens here, result stored for get_family())
        """
        self._detected_family = self._detect_family(settings)
        return True

    def get_priority(self) -> int:
        """Single detector, so priority doesn't matter."""
        return 1

    def get_family(self) -> ModelFamily:
        """Return the family detected in can_handle().

        Returns:
            ModelFamily determined during detection
        """
        return self._detected_family

    def _detect_family(self, settings: Any) -> ModelFamily:
        """Detect model family by importing class and checking inheritance.

        Pure inheritance-based detection - no string matching, no attributes.
        Uses Python's type system: issubclass() checks against base classes.

        Args:
            settings: Model configuration settings

        Returns:
            ModelFamily enum value based on class inheritance, or EXTERNAL if not found
        """
        # Try to get class_path directly first
        class_path = _string_attribute(settings, "class_path")

        # If no class_path, construct from name + module_path
        if not class_path:
            name = _string_attribute(settings, "name")
            module_path = _string_attribute(settings, "module_path")

            if name and module_path:
                # Construct full class path
                class_path = f"{module_path}.{name}"
            else:
                return ModelFamily.EXTERNAL

        if not class_path or "." not in class_path:
            return ModelFamily.EXTERNAL

        try:
            model_class = self._import_class(class_path)

            # Import base classes for checking
            from dlkit.domain.nn.graph.base import BaseGraphNetwork

            # Check inheritance (most specific first)
            if issubclass(model_class, BaseGraphNetwork):
                return ModelFamily.GRAPH

            if issubclass(model_class, LightningModule):
                return ModelFamily.EXTERNAL

            if issubclass(model_class, nn.Module):
                return ModelFamily.DLKIT_NN

            # Everything else is external
            return ModelFamily.EXTERNAL

        except Exception:
            return ModelFamily.EXTERNAL

    def _import_class(self, class_path: str) -> type:
        """Import a class from its fully qualified path.

        Args:
            class_path: Fully qualified class path (e.g., 'dlkit.domain.nn.ffnn.simple.FeedForwardNN')

        Returns:
            The imported class

        Raises:
            ImportError: If module or class cannot be imported
            ValueError: If class_path is malformed
        """
        if "." not in class_path:
            raise ValueError(f"Invalid class path (no module): {class_path}")

        import importlib

        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class ModelFamilyRegistry:
    """Registry for extensible model family detection.

    This registry maintains a collection of detectors and provides
    methods to register new detectors and detect model families.
    """

    def __init__(self):
        """Initialize registry with default detectors."""
        self._detectors: dict[ModelFamily, list[ModelFamilyDetector]] = {}
        self._register_default_detectors()

    def register_detector(self, detector: ModelFamilyDetector) -> None:
        """Register a new detector for a model family.

        Args:
            detector: Detector instance to register
        """
        family = detector.get_family()
        if family not in self._detectors:
            self._detectors[family] = []

        self._detectors[family].append(detector)
        # Sort by priority (lower numbers = higher priority)
        self._detectors[family].sort(key=lambda d: d.get_priority())

    def detect_family(self, settings: Any) -> ModelFamily:
        """Detect model family using registered detectors.

        Args:
            settings: Model configuration settings

        Returns:
            Detected ModelFamily enum value

        Raises:
            ValueError: If no detector can handle the settings
        """
        # Try detectors in priority order across all families
        all_detectors = []
        for detectors in self._detectors.values():
            all_detectors.extend(detectors)

        # Sort all detectors by priority
        all_detectors.sort(key=lambda d: d.get_priority())

        for detector in all_detectors:
            if detector.can_handle(settings):
                return detector.get_family()

        # If no detector matches, default to EXTERNAL
        return ModelFamily.EXTERNAL

    def get_detectors_for_family(self, family: ModelFamily) -> list[ModelFamilyDetector]:
        """Get all detectors registered for a specific family.

        Args:
            family: Model family to get detectors for

        Returns:
            List of detectors for the family (empty if none registered)
        """
        return self._detectors.get(family, []).copy()

    def get_all_families(self) -> set[ModelFamily]:
        """Get set of all registered model families.

        Returns:
            Set of ModelFamily enum values that have detectors
        """
        return set(self._detectors.keys())

    def _register_default_detectors(self) -> None:
        """Register default detector (single class-based detector).

        Uses ClassBasedDetector which inspects the actual model class's
        MODEL_FAMILY attribute instead of fragile string pattern matching.
        """
        self.register_detector(ClassBasedDetector())


class ModelFamilyRegistryFactory:
    """Factory for creating model family registries."""

    @staticmethod
    def create_default_registry() -> ModelFamilyRegistry:
        """Create registry with default detectors.

        Returns:
            ModelFamilyRegistry with all default detectors registered
        """
        return ModelFamilyRegistry()

    @staticmethod
    def create_custom_registry(detectors: list[ModelFamilyDetector]) -> ModelFamilyRegistry:
        """Create registry with custom detectors.

        Args:
            detectors: List of detectors to register

        Returns:
            ModelFamilyRegistry with specified detectors
        """
        registry = ModelFamilyRegistry()
        # Clear default detectors
        registry._detectors.clear()

        for detector in detectors:
            registry.register_detector(detector)

        return registry

    @staticmethod
    def create_minimal_registry() -> ModelFamilyRegistry:
        """Create registry with minimal detectors for testing.

        Returns:
            ModelFamilyRegistry with class-based detector
        """
        return ModelFamilyRegistryFactory.create_custom_registry(
            [
                ClassBasedDetector(),
            ]
        )
