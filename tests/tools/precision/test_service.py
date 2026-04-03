"""Tests for PrecisionService."""

from unittest.mock import Mock

import pytest
import torch

from dlkit.tools.precision import (
    PrecisionContext,
    PrecisionProvider,
    PrecisionService,
    PrecisionStrategy,
)


class TestPrecisionService:
    """Test suite for PrecisionService."""

    @pytest.fixture
    def precision_service(self):
        """Create a PrecisionService instance for testing."""
        # Create a service with a clean context
        context = PrecisionContext()
        context.clear_override()
        return PrecisionService(context)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock precision provider."""
        provider = Mock(spec=PrecisionProvider)
        provider.get_precision_strategy.return_value = PrecisionStrategy.MIXED_16
        return provider

    def test_resolve_precision_default(self, precision_service):
        """Test precision resolution with default fallback."""
        precision = precision_service.resolve_precision()
        assert precision == PrecisionStrategy.FULL_32  # Default

    def test_resolve_precision_with_provider(self, precision_service, mock_provider):
        """Test precision resolution with provider."""
        precision = precision_service.resolve_precision(mock_provider)
        assert precision == PrecisionStrategy.MIXED_16
        mock_provider.get_precision_strategy.assert_called_once()

    def test_resolve_precision_with_explicit_default(self, precision_service):
        """Test precision resolution with explicit default."""
        precision = precision_service.resolve_precision(default=PrecisionStrategy.TRUE_16)
        assert precision == PrecisionStrategy.TRUE_16

    def test_resolve_precision_with_context_override(self, precision_service):
        """Test that context override takes highest priority."""
        context = PrecisionContext()
        service = PrecisionService(context)

        # Set context override
        context.set_override(PrecisionStrategy.MIXED_BF16)

        # Override should take priority over provider and default
        mock_provider = Mock(spec=PrecisionProvider)
        mock_provider.get_precision_strategy.return_value = PrecisionStrategy.MIXED_16

        precision = service.resolve_precision(mock_provider, default=PrecisionStrategy.TRUE_16)
        assert precision == PrecisionStrategy.MIXED_BF16

        # Provider should not be called when context override exists
        mock_provider.get_precision_strategy.assert_not_called()

    def test_get_torch_dtype(self, precision_service, mock_provider):
        """Test torch.dtype resolution."""
        dtype = precision_service.get_torch_dtype(mock_provider)
        assert dtype == torch.float16  # MIXED_16 maps to float16

    def test_get_compute_dtype(self, precision_service, mock_provider):
        """Test compute dtype resolution."""
        compute_dtype = precision_service.get_compute_dtype(mock_provider)
        assert compute_dtype == torch.float32  # MIXED_16 uses float32 for computation

    def test_get_lightning_precision(self, precision_service, mock_provider):
        """Test Lightning precision parameter resolution."""
        lightning_precision = precision_service.get_lightning_precision(mock_provider)
        assert lightning_precision == "16-mixed"

    def test_is_mixed_precision(self, precision_service, mock_provider):
        """Test mixed precision detection."""
        is_mixed = precision_service.is_mixed_precision(mock_provider)
        assert is_mixed  # MIXED_16 is mixed precision

        # Test with non-mixed precision
        mock_provider.get_precision_strategy.return_value = PrecisionStrategy.FULL_32
        is_mixed = precision_service.is_mixed_precision(mock_provider)
        assert not is_mixed

    def test_cast_tensor(self, precision_service, mock_provider):
        """Test tensor casting."""
        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        casted_tensor = precision_service.cast_tensor(input_tensor, mock_provider)

        assert casted_tensor.dtype == torch.float16  # MIXED_16 target dtype
        assert torch.equal(casted_tensor, input_tensor.to(torch.float16))

    def test_apply_precision_to_model(self, precision_service, mock_provider):
        """Test model precision application."""
        model = torch.nn.Linear(10, 5)
        original_dtype = model.weight.dtype

        precision_model = precision_service.apply_precision_to_model(model, mock_provider)

        # Model should be cast to target precision
        assert precision_model.weight.dtype == torch.float16
        assert precision_model.weight.dtype != original_dtype
        assert precision_model is model  # Should return same instance

    def test_get_precision_info(self, precision_service, mock_provider):
        """Test precision information gathering."""
        info = precision_service.get_precision_info(mock_provider)

        expected_keys = {
            "strategy",
            "value",
            "torch_dtype",
            "compute_dtype",
            "lightning_precision",
            "supports_autocast",
            "is_reduced_precision",
            "memory_factor",
            "context_override",
        }
        assert set(info.keys()) == expected_keys

        assert info["strategy"] == "MIXED_16"
        assert info["value"] == "16-mixed"
        assert info["torch_dtype"] == "torch.float16"
        assert info["compute_dtype"] == "torch.float32"
        assert info["lightning_precision"] == "16-mixed"
        assert info["supports_autocast"] is True
        assert info["is_reduced_precision"] is True
        assert info["memory_factor"] == 0.7
        assert info["context_override"] is None

    def test_provider_without_precision_method(self, precision_service):
        """Test handling of provider without precision method."""

        # Create a provider that doesn't implement get_precision_strategy
        class InvalidProvider:
            pass

        invalid_provider = InvalidProvider()

        # Should fall back to default
        precision = precision_service.resolve_precision(invalid_provider)
        assert precision == PrecisionStrategy.FULL_32

    def test_string_representation(self, precision_service):
        """Test string representation."""
        repr_str = repr(precision_service)
        assert "PrecisionService" in repr_str
        assert "context_override=None" in repr_str
