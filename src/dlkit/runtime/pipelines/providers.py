"""Data providers for loading and managing dataflow from different sources.

This module implements the DataProvider interface for various dataflow sources,
focusing on file-based dataflow (Features and Targets). Providers handle dataflow
loading and caching only. Transform application is handled by the
processing pipeline (TransformApplicationStep).
"""

import torch

from dlkit.tools.io.arrays import load_array
from dlkit.tools.config.data_entries import (
    DataEntry,
    PathBasedEntry,
    ValueBasedEntry,
    PathFeature,
    PathTarget,
    ValueFeature,
    ValueTarget,
    is_feature_entry,
    is_target_entry,
)
from .interfaces import DataProvider


class FileDataProvider(DataProvider):
    """Provider for file-based dataflow (Features and Targets).

    Loads arrays/tensors from disk and caches them in memory. Transform application
    is handled by the processing pipeline, not by the provider.

    Attributes:
        _cache (dict[str, torch.Tensor]): Internal cache for loaded tensors.
        _raw_cache (dict[str, torch.Tensor]): Raw cache for direct file loads.
    """

    def __init__(self):
        """Initialize the file dataflow provider with empty caches."""
        self._cache: dict[str, torch.Tensor] = {}
        self._raw_cache: dict[str, torch.Tensor] = {}

    def can_handle(self, entry: DataEntry) -> bool:
        """Check if this provider can handle the given dataflow entry type.

        This provider handles:
        - PathBasedEntry instances (PathFeature, PathTarget) with valid paths
        - ValueBasedEntry instances (ValueFeature, ValueTarget) with in-memory values
        - Legacy Feature/Target with has_path() returning True

        It cannot handle latent entries, predictions, or placeholder entries.

        Args:
            entry (DataEntry): Data entry configuration to check.

        Returns:
            bool: True if entry has data (path or value), False otherwise.
        """
        # New hierarchy: PathBasedEntry with valid path
        if isinstance(entry, PathBasedEntry):
            return entry.has_path()  # False for placeholders
        # New hierarchy: ValueBasedEntry always has value
        if isinstance(entry, ValueBasedEntry):
            return entry.has_value()  # False for placeholders
        # Legacy fallback: any entry with has_path() or has_value()
        return entry.has_path() or entry.has_value()

    def load_data(self, entry: DataEntry, idx: int) -> torch.Tensor:
        """Load dataflow for a specific entry and index.

        Implements lazy loading with caching. Data is loaded once and cached for
        subsequent access.

        Args:
            entry (DataEntry): Data entry configuration (must be Feature or Target).
            idx (int): Index of the dataflow sample to load.

        Returns:
            torch.Tensor: Loaded tensor for the specified index.

        Raises:
            ValueError: If entry type cannot be handled.
            RuntimeError: If dataflow loading fails.
        """
        if not self.can_handle(entry):
            raise ValueError(f"Cannot handle entry type: {type(entry)}")

        # Load and cache dataflow if not already cached
        if entry.name not in self._cache:
            self._load_and_transform_data(entry)

        return self._cache[entry.name][idx]

    def get_length(self, entry: DataEntry) -> int:
        """Get the total length of dataflow for this entry.

        Args:
            entry (DataEntry): Data entry configuration.

        Returns:
            int: Total number of samples available.

        Raises:
            ValueError: If entry type cannot be handled.
        """
        if not self.can_handle(entry):
            raise ValueError(f"Cannot handle entry type: {type(entry)}")

        # Load raw dataflow to determine length if not cached
        if entry.name not in self._raw_cache:
            self._load_raw_data(entry)

        return len(self._raw_cache[entry.name])

    def clear_cache(self) -> None:
        """Clear all cached dataflow to free memory."""
        self._cache.clear()
        self._raw_cache.clear()

    def _load_and_transform_data(self, entry: DataEntry) -> None:
        """Load raw dataflow and apply transforms, caching the result.

        Args:
            entry: Data entry to load and transform (PathBasedEntry or ValueBasedEntry)
        """
        # Load raw dataflow if not already cached
        if entry.name not in self._raw_cache:
            self._load_raw_data(entry)

        # Do not apply transforms here (handled by processing pipeline)
        # Cache raw dataflow directly as the accessed tensor source
        self._cache[entry.name] = self._raw_cache[entry.name]

    def _load_raw_data(self, entry: DataEntry) -> None:
        """Load raw dataflow from file or in-memory value and cache it.

        Args:
            entry: Data entry to load (PathBasedEntry with path or ValueBasedEntry with value)

        Raises:
            RuntimeError: If file loading fails
            ValueError: If entry has neither path nor value
        """
        try:
            # ValueBasedEntry: convert in-memory value to tensor
            if isinstance(entry, ValueBasedEntry) and entry.has_value():
                raw_data = torch.as_tensor(entry.value)
                if entry.dtype is not None:
                    raw_data = raw_data.to(dtype=entry.dtype)
                self._raw_cache[entry.name] = raw_data
            # PathBasedEntry: load from file
            elif isinstance(entry, PathBasedEntry) and entry.has_path() and entry.path is not None:
                raw_data = load_array(entry.path, dtype=entry.dtype)
                self._raw_cache[entry.name] = raw_data
            # Legacy fallback: check has_value/has_path methods
            elif entry.has_value():
                raw_data = torch.as_tensor(getattr(entry, "value"))
                if entry.dtype is not None:
                    raw_data = raw_data.to(dtype=entry.dtype)
                self._raw_cache[entry.name] = raw_data
            elif entry.has_path():
                raw_data = load_array(getattr(entry, "path"), dtype=entry.dtype)
                self._raw_cache[entry.name] = raw_data
            else:
                raise ValueError(f"Entry '{entry.name}' has neither path nor value")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to load dataflow for {entry.name}: {e}") from e

class ProviderRegistry:
    """Registry for managing dataflow providers using Registry Pattern.

    This registry routes dataflow entries to appropriate providers and
    manages the collection of available providers. It follows the
    Open/Closed principle by allowing new providers to be added
    without modifying existing code.

    Attributes:
        _providers: List of registered dataflow providers
    """

    def __init__(self, providers: list[DataProvider] | None = None):
        """Initialize the provider registry.

        Args:
            providers: Initial list of providers to register
        """
        self._providers: list[DataProvider] = providers or []

    def get_provider(self, entry: DataEntry) -> DataProvider:
        """Route entry to the appropriate provider.

        Args:
            entry: Data entry to find a provider for

        Returns:
            Provider that can handle the entry

        Raises:
            ValueError: If no provider can handle the entry
        """
        for provider in self._providers:
            if provider.can_handle(entry):
                return provider

        raise ValueError(f"No provider found for entry type: {type(entry).__name__}")

    def add_provider(self, provider: DataProvider) -> None:
        """Add a new provider to the registry.

        Args:
            provider: Provider to add to the registry
        """
        self._providers.append(provider)

    def remove_provider(self, provider: DataProvider) -> bool:
        """Remove a provider from the registry.

        Args:
            provider: Provider to remove

        Returns:
            True if provider was found and removed, False otherwise
        """
        try:
            self._providers.remove(provider)
            return True
        except ValueError:
            return False

    def get_providers_for_entry(self, entry: DataEntry) -> list[DataProvider]:
        """Get all providers that can handle a given entry.

        Args:
            entry: Data entry to find providers for

        Returns:
            List of providers that can handle the entry
        """
        return [provider for provider in self._providers if provider.can_handle(entry)]
