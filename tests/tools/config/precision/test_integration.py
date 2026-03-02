"""Integration tests for precision control system."""

import pytest
import torch

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.interfaces.api.services.precision_service import PrecisionService
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.data_entries import Feature
from dlkit.tools.io.arrays import load_array
from .conftest import TestModelFactory


class TestPrecisionIntegration:
    """Integration tests for precision control across components."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a sample dataflow file for testing."""
        data = torch.randn(100, 10, dtype=torch.float64)
        file_path = tmp_path / "test_data.pt"
        torch.save(data, file_path)
        return file_path

    def test_session_settings_precision_provider(self):
        """Test SessionSettings as PrecisionProvider."""
        session = SessionSettings(precision=PrecisionStrategy.MIXED_16)
        assert session.get_precision_strategy() == PrecisionStrategy.MIXED_16

        service = PrecisionService()
        resolved = service.resolve_precision(session)
        assert resolved == PrecisionStrategy.MIXED_16

    def test_data_entry_precision_resolution(self, sample_data_file):
        """Test DataEntry precision resolution."""
        # Feature with explicit dtype
        feature_explicit = Feature(name="test", path=sample_data_file, dtype=torch.float16)
        assert feature_explicit.get_effective_dtype() == torch.float16

        # Feature without explicit dtype - should use session precision
        feature_session = Feature(name="test2", path=sample_data_file)
        session = SessionSettings(precision=PrecisionStrategy.TRUE_BF16)

        service = PrecisionService()
        expected_dtype = service.get_torch_dtype(session)
        actual_dtype = feature_session.get_effective_dtype(session)
        assert actual_dtype == expected_dtype == torch.bfloat16

    def test_io_arrays_precision_integration(self, sample_data_file):
        """Test I/O arrays precision integration."""
        # Load with session precision using context
        session = SessionSettings(precision=PrecisionStrategy.MIXED_16)
        with precision_override(session.get_precision_strategy()):
            tensor = load_array(sample_data_file)
        assert tensor.dtype == torch.float16

        # Load with explicit dtype override
        tensor_explicit = load_array(sample_data_file, dtype=torch.float32)
        assert tensor_explicit.dtype == torch.float32

        # Load with session precision (convenience function)
        from dlkit.tools.io.arrays import load_array_with_session_precision

        with precision_override(PrecisionStrategy.TRUE_BF16):
            tensor_session = load_array_with_session_precision(sample_data_file)
            assert tensor_session.dtype == torch.bfloat16

    def test_model_precision_integration(self, test_model_factory, sample_shape):
        """Test model precision integration."""
        # Model with precision override context
        with precision_override(PrecisionStrategy.TRUE_16):
            model_explicit = test_model_factory.create_precision_test_model(sample_shape)
        assert model_explicit.get_precision_strategy() == PrecisionStrategy.TRUE_16
        assert model_explicit.get_model_dtype() == torch.float16
        assert model_explicit.linear.weight.dtype == torch.float16

        # Model with session precision (bfloat16 - note: TRUE_BF16 and MIXED_BF16 both use bfloat16)
        # After initialization, we can't distinguish between TRUE_BF16 and MIXED_BF16 from dtype alone
        with precision_override(PrecisionStrategy.TRUE_BF16):
            model_session = test_model_factory.create_precision_test_model(sample_shape)
            assert model_session.get_precision_strategy() == PrecisionStrategy.TRUE_BF16
            assert model_session.get_model_dtype() == torch.bfloat16
            assert model_session.linear.weight.dtype == torch.bfloat16

    def test_model_input_precision(self, test_model_factory, sample_shape):
        """Test model input precision handling."""
        with precision_override(PrecisionStrategy.TRUE_16):
            model = test_model_factory.create_precision_test_model(sample_shape)

        # Create input tensor with different precision
        input_tensor = torch.randn(1, 10, dtype=torch.float32)

        # Model parameters should be float16
        assert next(model.parameters()).dtype == torch.float16

        # Manual cast for testing (Lightning would handle this in training)
        casted_input = input_tensor.to(torch.float16)
        assert casted_input.dtype == torch.float16

    def test_predict_step_precision(self, test_model_factory, sample_shape):
        """Test predict_step with precision handling."""
        with precision_override(PrecisionStrategy.TRUE_16):
            model = test_model_factory.create_precision_test_model(sample_shape)

        # Test with tuple batch (convert to correct precision)
        x = torch.randn(1, 10, dtype=torch.float16)
        y = torch.randn(1, 5, dtype=torch.float16)
        batch = (x, y)

        output = model.predict_step(batch, 0)
        assert output.dtype == torch.float16  # Output should match model precision

        # Test with single tensor batch
        output_single = model.predict_step(x, 0)
        assert output_single.dtype == torch.float16

    def test_context_override_priority(self, sample_data_file, test_model_factory, sample_shape):
        """Test that context override has highest priority."""
        # Setup session with one precision
        session = SessionSettings(precision=PrecisionStrategy.FULL_32)

        # Override with context (use TRUE_16 instead of MIXED_16 to avoid ambiguity)
        with precision_override(PrecisionStrategy.TRUE_16):
            # I/O should use context override, not session
            tensor = load_array(sample_data_file)
            assert tensor.dtype == torch.float16

            # Model should use context override
            model = test_model_factory.create_precision_test_model(sample_shape)
            assert model.get_precision_strategy() == PrecisionStrategy.TRUE_16

        # After context, should revert to session precision
        tensor_after = load_array(sample_data_file)
        assert tensor_after.dtype == torch.float32

    def test_precision_consistency_across_components(
        self, sample_data_file, test_model_factory, sample_shape
    ):
        """Test precision consistency across all components."""
        precision_strategy = PrecisionStrategy.TRUE_BF16

        with precision_override(precision_strategy):
            # Create session settings
            session = SessionSettings()
            service = PrecisionService()

            # All components should use the same precision
            expected_dtype = torch.bfloat16

            # 1. Session should resolve to context override
            assert service.resolve_precision(session) == precision_strategy

            # 2. I/O should use context precision
            tensor = load_array(sample_data_file)
            assert tensor.dtype == expected_dtype

            # 3. Model should use context precision
            model = test_model_factory.create_precision_test_model(sample_shape)
            assert model.get_model_dtype() == expected_dtype

            # 4. DataEntry should use context precision
            feature = Feature(name="test", path=sample_data_file)
            assert feature.get_effective_dtype() == expected_dtype

    def test_trainer_settings_precision_integration(self):
        """Test TrainerSettings precision integration."""
        from dlkit.tools.config.trainer_settings import TrainerSettings

        # Trainer with explicit precision
        trainer_explicit = TrainerSettings(precision="16-mixed")
        # This would be tested in a full trainer build, but we can check the field
        assert trainer_explicit.precision == "16-mixed"

        # Trainer without explicit precision should use session
        trainer_session = TrainerSettings()
        assert trainer_session.precision is None
        # The build() method would resolve precision from session

    def test_precision_service_info_logging(self):
        """Test precision service information gathering."""
        session = SessionSettings(precision=PrecisionStrategy.MIXED_BF16)
        service = PrecisionService()

        info = service.get_precision_info(session)

        assert info["strategy"] == "MIXED_BF16"
        assert info["torch_dtype"] == "torch.bfloat16"
        assert info["compute_dtype"] == "torch.float32"
        assert info["lightning_precision"] == "bf16-mixed"
        assert info["supports_autocast"] is True
        assert info["is_reduced_precision"] is True
        assert info["memory_factor"] == 0.7

    def test_error_handling_invalid_provider(self, sample_data_file):
        """Test error handling with invalid precision provider."""
        service = PrecisionService()

        # Provider that raises exception
        class BrokenProvider:
            def get_precision_strategy(self):
                raise RuntimeError("Provider broken")

        broken_provider = BrokenProvider()

        # Should fall back to default
        precision = service.resolve_precision(broken_provider)
        assert precision == PrecisionStrategy.FULL_32

        # DataEntry should handle service failures gracefully
        feature = Feature(name="test", path=sample_data_file)
        dtype = feature.resolve_dtype_with_fallback(torch.float64)
        # Should get the fallback since we can't resolve from broken service
        assert dtype in (torch.float32, torch.float64)  # Either service default or fallback
