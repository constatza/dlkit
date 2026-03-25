"""Data entry registry for exposing data entries to end users.

This module provides a singleton registry that stores and exposes data entries
(features, targets, latents, predictions) to end users, fulfilling the requirement
that data entries should be accessible after dataset creation.
"""

from __future__ import annotations

from threading import Lock
from typing import Any

from dlkit.tools.config.data_entries import (
    DataEntry,
    Latent,
    PathFeature,
    PathTarget,
    Prediction,
    SparseFeature,
    ValueFeature,
    ValueTarget,
)

FeatureEntry = PathFeature | ValueFeature | SparseFeature
TargetEntry = PathTarget | ValueTarget


class DataEntryRegistry:
    """Singleton registry for storing and exposing data entries to end users.

    This registry provides a centralized location where data entries can be
    stored during wrapper initialization and accessed by end users throughout
    the application lifecycle.

    Thread-safe singleton implementation ensures consistent access across
    the application.

    Example:
        ```python
        # During wrapper initialization
        registry = DataEntryRegistry.get_instance()
        registry.register_entries(entry_configs)

        # Later, in user code
        registry = DataEntryRegistry.get_instance()
        features = registry.get_features()
        targets = registry.get_targets()
        ```
    """

    _instance: DataEntryRegistry | None = None
    _lock = Lock()

    def __init__(self) -> None:
        """Initialize empty registry."""
        if DataEntryRegistry._instance is not None:
            raise RuntimeError("DataEntryRegistry is a singleton. Use get_instance().")
        self._entries: dict[str, DataEntry] = {}
        self._lock = Lock()

    @classmethod
    def get_instance(cls) -> DataEntryRegistry:
        """Get the singleton instance, creating it if necessary.

        Returns:
            DataEntryRegistry: The singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_entries(self, entry_configs: dict[str, DataEntry]) -> None:
        """Register data entries in the registry.

        Args:
            entry_configs: Dictionary mapping entry names to DataEntry objects
        """
        with self._lock:
            self._entries.update(entry_configs)

    def get_entries(self) -> dict[str, DataEntry]:
        """Get all registered data entries.

        Returns:
            Dictionary mapping entry names to DataEntry objects
        """
        with self._lock:
            return self._entries.copy()

    def get_features(self) -> dict[str, FeatureEntry]:
        """Get all feature entries.

        Returns:
            Dictionary mapping feature names to feature objects
        """
        with self._lock:
            return {
                name: entry
                for name, entry in self._entries.items()
                if isinstance(entry, (PathFeature, ValueFeature, SparseFeature))
            }

    def get_targets(self) -> dict[str, TargetEntry]:
        """Get all target entries.

        Returns:
            Dictionary mapping target names to target objects
        """
        with self._lock:
            return {
                name: entry
                for name, entry in self._entries.items()
                if isinstance(entry, (PathTarget, ValueTarget))
            }

    def get_latents(self) -> dict[str, Latent]:
        """Get all latent entries.

        Returns:
            Dictionary mapping latent names to Latent objects
        """
        with self._lock:
            return {
                name: entry for name, entry in self._entries.items() if isinstance(entry, Latent)
            }

    def get_predictions(self) -> dict[str, Prediction]:
        """Get all prediction entries.

        Returns:
            Dictionary mapping prediction names to Prediction objects
        """
        with self._lock:
            return {
                name: entry
                for name, entry in self._entries.items()
                if isinstance(entry, Prediction)
            }

    def get_entry(self, name: str) -> DataEntry | None:
        """Get a specific entry by name.

        Args:
            name: Name of the entry to retrieve

        Returns:
            DataEntry object if found, None otherwise
        """
        with self._lock:
            return self._entries.get(name)

    def clear(self) -> None:
        """Clear all registered entries.

        Primarily for testing purposes.
        """
        with self._lock:
            self._entries.clear()

    def get_entry_info(self) -> dict[str, Any]:
        """Get summary information about registered entries.

        Returns:
            Dictionary with entry counts and names by type
        """
        with self._lock:
            features = self.get_features()
            targets = self.get_targets()
            latents = self.get_latents()
            predictions = self.get_predictions()

            return {
                "total_entries": len(self._entries),
                "feature_count": len(features),
                "target_count": len(targets),
                "latent_count": len(latents),
                "prediction_count": len(predictions),
                "feature_names": list(features.keys()),
                "target_names": list(targets.keys()),
                "latent_names": list(latents.keys()),
                "prediction_names": list(predictions.keys()),
            }
