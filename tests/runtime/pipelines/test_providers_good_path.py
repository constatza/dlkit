"""Good-path tests for dataflow providers following SOLID principles.

This module tests the primary behaviors of FileDataProvider and ProviderRegistry
focusing on successful execution paths. Error scenarios are tested separately
to maintain single responsibility per test function.
"""

from __future__ import annotations

import torch
import pytest

from dlkit.runtime.pipelines.providers import FileDataProvider, ProviderRegistry
from dlkit.runtime.pipelines.interfaces import DataProvider
from dlkit.tools.config.data_entries import Feature, Target, Latent, Prediction


class TestFileDataProviderCanHandle:
    """Test FileDataProvider.can_handle() method for entry type validation."""

    def test_can_handle_feature_entry(
        self, file_data_provider: FileDataProvider, feature_entry: Feature
    ) -> None:
        """Test that provider correctly identifies Feature entries."""
        assert file_data_provider.can_handle(feature_entry) is True

    def test_can_handle_target_entry(
        self, file_data_provider: FileDataProvider, target_entry: Target
    ) -> None:
        """Test that provider correctly identifies Target entries."""
        assert file_data_provider.can_handle(target_entry) is True

    def test_cannot_handle_latent_entry(
        self, file_data_provider: FileDataProvider, latent_entry: Latent
    ) -> None:
        """Test that provider rejects Latent entries (no file path)."""
        assert file_data_provider.can_handle(latent_entry) is False

    def test_cannot_handle_prediction_entry(
        self, file_data_provider: FileDataProvider, prediction_entry: Prediction
    ) -> None:
        """Test that provider rejects Prediction entries."""
        assert file_data_provider.can_handle(prediction_entry) is False


class TestFileDataProviderDataLoading:
    """Test FileDataProvider dataflow loading and caching behavior."""

    def test_load_data_feature_entry(
        self,
        file_data_provider: FileDataProvider,
        feature_entry: Feature,
        sample_numpy_data: torch.Tensor,
    ) -> None:
        """Test loading dataflow from Feature entry with correct indexing."""
        # Load first sample
        loaded_data = file_data_provider.load_data(feature_entry, 0)

        # Verify shape and type
        assert isinstance(loaded_data, torch.Tensor)
        assert loaded_data.dtype == feature_entry.dtype
        assert loaded_data.shape == sample_numpy_data[0].shape

        # Verify dataflow is cached (second load should be from cache)
        cached_data = file_data_provider.load_data(feature_entry, 0)
        assert torch.equal(loaded_data, cached_data)

    def test_load_data_target_entry(
        self,
        file_data_provider: FileDataProvider,
        target_entry: Target,
        sample_torch_data: torch.Tensor,
    ) -> None:
        """Test loading dataflow from Target entry (.pt file)."""
        # Load middle sample
        loaded_data = file_data_provider.load_data(target_entry, 5)

        # Verify shape and type
        assert isinstance(loaded_data, torch.Tensor)
        assert loaded_data.dtype == target_entry.dtype
        assert loaded_data.shape == sample_torch_data[5].shape

    def test_load_data_different_indices(
        self, file_data_provider: FileDataProvider, feature_entry: Feature
    ) -> None:
        """Test loading dataflow from different indices returns different values."""
        data_0 = file_data_provider.load_data(feature_entry, 0)
        data_1 = file_data_provider.load_data(feature_entry, 1)

        # Different indices should return different dataflow (with high probability)
        assert not torch.equal(data_0, data_1)

    def test_lazy_loading_with_caching(
        self, file_data_provider: FileDataProvider, feature_entry: Feature
    ) -> None:
        """Test that dataflow is loaded once and cached for subsequent access."""
        # Initially no dataflow in cache
        assert feature_entry.name not in file_data_provider._cache
        assert feature_entry.name not in file_data_provider._raw_cache

        # First load triggers caching
        file_data_provider.load_data(feature_entry, 0)
        assert feature_entry.name in file_data_provider._cache
        assert feature_entry.name in file_data_provider._raw_cache

        # Cache should contain the loaded tensor
        cached_tensor = file_data_provider._cache[feature_entry.name]
        assert isinstance(cached_tensor, torch.Tensor)


class TestFileDataProviderLength:
    """Test FileDataProvider.get_length() method."""

    def test_get_length_feature_entry(
        self,
        file_data_provider: FileDataProvider,
        feature_entry: Feature,
        sample_numpy_data: torch.Tensor,
    ) -> None:
        """Test getting correct length for Feature entry."""
        length = file_data_provider.get_length(feature_entry)
        assert length == len(sample_numpy_data)

    def test_get_length_target_entry(
        self,
        file_data_provider: FileDataProvider,
        target_entry: Target,
        sample_torch_data: torch.Tensor,
    ) -> None:
        """Test getting correct length for Target entry."""
        length = file_data_provider.get_length(target_entry)
        assert length == len(sample_torch_data)

    def test_get_length_caches_raw_data(
        self, file_data_provider: FileDataProvider, feature_entry: Feature
    ) -> None:
        """Test that get_length() caches raw dataflow for efficiency."""
        # Initially no dataflow in raw cache
        assert feature_entry.name not in file_data_provider._raw_cache

        # get_length should populate raw cache
        file_data_provider.get_length(feature_entry)
        assert feature_entry.name in file_data_provider._raw_cache


class TestFileDataProviderCacheManagement:
    """Test FileDataProvider cache management functionality."""

    def test_clear_cache_empties_all_caches(
        self, populated_provider: FileDataProvider, feature_entry: Feature, target_entry: Target
    ) -> None:
        """Test that clear_cache() removes all cached"""
        # Verify dataflow is cached
        assert feature_entry.name in populated_provider._raw_cache
        assert target_entry.name in populated_provider._raw_cache

        # Clear cache
        populated_provider.clear_cache()

        # Verify caches are empty
        assert len(populated_provider._cache) == 0
        assert len(populated_provider._raw_cache) == 0

    def test_cache_consistency_after_clear(
        self, file_data_provider: FileDataProvider, feature_entry: Feature
    ) -> None:
        """Test that dataflow can be reloaded after cache clear."""
        # Load dataflow and cache it
        original_data = file_data_provider.load_data(feature_entry, 0)

        # Clear cache and reload
        file_data_provider.clear_cache()
        reloaded_data = file_data_provider.load_data(feature_entry, 0)

        # Data should be identical after reload
        assert torch.equal(original_data, reloaded_data)


class TestFileDataProviderTransforms:
    """Test FileDataProvider transform handling (backward compatibility)."""

    def test_apply_transforms_no_op(
        self,
        file_data_provider: FileDataProvider,
        sample_torch_data: torch.Tensor,
        transform_settings: list,
    ) -> None:
        """Test that _apply_transforms returns dataflow unchanged (no-op)."""
        result = file_data_provider._apply_transforms(sample_torch_data, transform_settings)
        assert torch.equal(result, sample_torch_data)


class TestFileDataProviderErrors:
    """Test FileDataProvider error handling scenarios."""

    def test_load_data_unsupported_entry_type(
        self, file_data_provider: FileDataProvider, latent_entry: Latent
    ) -> None:
        """Test that loading unsupported entry type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot handle entry type"):
            file_data_provider.load_data(latent_entry, 0)

    def test_get_length_unsupported_entry_type(
        self, file_data_provider: FileDataProvider, prediction_entry: Prediction
    ) -> None:
        """Test that getting length of unsupported entry type raises ValueError."""
        with pytest.raises(ValueError, match="Cannot handle entry type"):
            file_data_provider.get_length(prediction_entry)

    def test_load_data_nonexistent_file(
        self, file_data_provider: FileDataProvider, invalid_feature_entry: Feature
    ) -> None:
        """Test that loading from non-existent file raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to load dataflow"):
            file_data_provider.load_data(invalid_feature_entry, 0)

    def test_get_length_nonexistent_file(
        self, file_data_provider: FileDataProvider, invalid_feature_entry: Feature
    ) -> None:
        """Test that getting length of non-existent file raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Failed to load dataflow"):
            file_data_provider.get_length(invalid_feature_entry)


class TestProviderRegistryBasic:
    """Test basic ProviderRegistry functionality."""

    def test_registry_initialization_empty(self) -> None:
        """Test creating registry with no initial providers."""
        registry = ProviderRegistry()
        assert len(registry._providers) == 0

    def test_registry_initialization_with_providers(
        self, multi_provider_setup: list[DataProvider]
    ) -> None:
        """Test creating registry with initial providers."""
        registry = ProviderRegistry(multi_provider_setup)
        assert len(registry._providers) == len(multi_provider_setup)
        assert all(provider in registry._providers for provider in multi_provider_setup)

    def test_add_provider(self, file_data_provider: FileDataProvider) -> None:
        """Test adding a provider to the registry."""
        registry = ProviderRegistry()
        initial_count = len(registry._providers)

        registry.add_provider(file_data_provider)

        assert len(registry._providers) == initial_count + 1
        assert file_data_provider in registry._providers

    def test_remove_provider_success(self, file_data_provider: FileDataProvider) -> None:
        """Test successful removal of a provider from the registry."""
        registry = ProviderRegistry([file_data_provider])

        result = registry.remove_provider(file_data_provider)

        assert result is True
        assert file_data_provider not in registry._providers

    def test_remove_provider_not_found(
        self, file_data_provider: FileDataProvider, mock_provider: DataProvider
    ) -> None:
        """Test removing a provider that's not in the registry."""
        registry = ProviderRegistry([file_data_provider])

        result = registry.remove_provider(mock_provider)

        assert result is False
        assert file_data_provider in registry._providers  # Original provider unchanged


class TestProviderRegistryRouting:
    """Test ProviderRegistry provider routing functionality."""

    def test_get_provider_feature_entry(
        self, feature_entry: Feature, multi_provider_setup: list[DataProvider]
    ) -> None:
        """Test routing Feature entry to appropriate provider."""
        registry = ProviderRegistry(multi_provider_setup)

        provider = registry.get_provider(feature_entry)

        assert provider.can_handle(feature_entry)
        assert isinstance(provider.load_data(feature_entry, 0), torch.Tensor)

    def test_get_provider_target_entry(
        self, target_entry: Target, multi_provider_setup: list[DataProvider]
    ) -> None:
        """Test routing Target entry to appropriate provider."""
        registry = ProviderRegistry(multi_provider_setup)

        provider = registry.get_provider(target_entry)

        assert provider.can_handle(target_entry)
        assert isinstance(provider.load_data(target_entry, 0), torch.Tensor)

    def test_get_provider_no_handler(
        self, latent_entry: Latent, multi_provider_setup: list[DataProvider]
    ) -> None:
        """Test error when no provider can handle the entry."""
        registry = ProviderRegistry(multi_provider_setup)

        with pytest.raises(ValueError, match="No provider found for entry type"):
            registry.get_provider(latent_entry)

    def test_get_providers_for_entry_multiple_matches(self, feature_entry: Feature) -> None:
        """Test getting all providers that can handle a specific entry."""
        # Create multiple providers that can handle Feature entries
        provider1 = FileDataProvider()
        provider2 = FileDataProvider()

        registry = ProviderRegistry([provider1, provider2])

        matching_providers = registry.get_providers_for_entry(feature_entry)

        assert len(matching_providers) == 2
        assert provider1 in matching_providers
        assert provider2 in matching_providers

    def test_get_providers_for_entry_no_matches(
        self, latent_entry: Latent, multi_provider_setup: list[DataProvider]
    ) -> None:
        """Test getting providers when none can handle the entry."""
        registry = ProviderRegistry(multi_provider_setup)

        matching_providers = registry.get_providers_for_entry(latent_entry)

        assert len(matching_providers) == 0


class TestProviderRegistryIntegration:
    """Test ProviderRegistry with FileDataProvider integration."""

    def test_end_to_end_feature_loading(
        self, feature_entry: Feature, sample_numpy_data: torch.Tensor
    ) -> None:
        """Test complete workflow: registry routing to provider and dataflow loading."""
        provider = FileDataProvider()
        registry = ProviderRegistry([provider])

        # Get provider through registry and load dataflow
        selected_provider = registry.get_provider(feature_entry)
        loaded_data = selected_provider.load_data(feature_entry, 0)

        # Verify correct dataflow loading
        assert isinstance(loaded_data, torch.Tensor)
        assert loaded_data.dtype == feature_entry.dtype

    def test_registry_provider_isolation(
        self, feature_entry: Feature, target_entry: Target
    ) -> None:
        """Test that providers in registry maintain independent caches."""
        provider1 = FileDataProvider()
        provider2 = FileDataProvider()
        registry = ProviderRegistry([provider1, provider2])

        # Load dataflow through first provider
        provider1.load_data(feature_entry, 0)
        assert registry.get_provider(feature_entry) in {provider1, provider2}

        # Second provider should have empty cache
        assert feature_entry.name not in provider2._cache
        assert feature_entry.name in provider1._cache
