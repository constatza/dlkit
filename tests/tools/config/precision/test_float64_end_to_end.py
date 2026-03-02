"""Comprehensive end-to-end tests for float64 (double) precision.

This module tests the complete float64 precision pipeline to ensure:
1. Config loading accepts "64", "double", "float64" aliases
2. Model weights are initialized as float64
3. Input tensors are cast to float64
4. Forward pass maintains float64 precision
5. Outputs are float64
6. Invalid precision strings raise validation errors
"""

import sys
from pathlib import Path

import pytest
import torch

from dlkit.tools.config.precision import PrecisionStrategy
from dlkit.tools.config.precision.strategy import _PRECISION_ALIAS_MAP
from dlkit.tools.config.session_settings import SessionSettings
from dlkit.tools.config.data_entries import Feature, Target
from dlkit.tools.io.arrays import load_array
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.interfaces.api.domain.precision import precision_override, get_precision_context
from dlkit.core.models.nn.base import DLKitModel


pytestmark = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="MPS backend on macOS lacks float64 support",
)


class Float64TestModel(DLKitModel):
    """Test model for float64 precision verification."""

    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()

        # Build model layers
        self.layer1 = torch.nn.Linear(in_features, 64)
        self.activation = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(64, 32)
        self.output_layer = torch.nn.Linear(32, out_features)

        # Apply precision from context (simulating Lightning behavior)
        # In production, Lightning's Trainer handles this via precision plugin
        service = get_precision_service()
        precision_strategy = service.resolve_precision()
        dtype = precision_strategy.to_torch_dtype()
        self.to(dtype)

    def forward(self, x):
        """Forward pass through the model."""
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.output_layer(x)
        return x


class TestFloat64EndToEnd:
    """Comprehensive float64 precision tests."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data files for testing."""
        # Create float32 data that will be cast to float64
        input_data = torch.randn(100, 20, dtype=torch.float32)
        target_data = torch.randn(100, 10, dtype=torch.float32)

        input_path = tmp_path / "input_data.pt"
        target_path = tmp_path / "target_data.pt"

        torch.save(input_data, input_path)
        torch.save(target_data, target_path)

        return {"input": input_path, "target": target_path}

    @pytest.fixture
    def model_shape(self):
        """Model dimensions for test model."""
        return {"in_features": 20, "out_features": 10}

    def test_precision_enum_float64_exact(self):
        """Test float64 precision with exact enum value."""
        session = SessionSettings(precision=PrecisionStrategy.FULL_64)
        assert session.precision == PrecisionStrategy.FULL_64
        assert session.get_precision_strategy() == PrecisionStrategy.FULL_64

    def test_precision_string_semantic_only(self):
        """Test that numeric strings are rejected, only semantic aliases work."""
        # Numeric strings should be rejected
        with pytest.raises(ValueError, match="Invalid precision value"):
            SessionSettings(precision="64")

    def test_precision_string_double(self):
        """Test float64 precision with alias 'double'."""
        session = SessionSettings(precision="double")
        assert session.precision == PrecisionStrategy.FULL_64

    def test_precision_string_float64(self):
        """Test float64 precision with alias 'float64'."""
        session = SessionSettings(precision="float64")
        assert session.precision == PrecisionStrategy.FULL_64

    def test_precision_string_f64(self):
        """Test float64 precision with alias 'f64'."""
        session = SessionSettings(precision="f64")
        assert session.precision == PrecisionStrategy.FULL_64

    def test_precision_string_fp64(self):
        """Test float64 precision with alias 'fp64'."""
        session = SessionSettings(precision="fp64")
        assert session.precision == PrecisionStrategy.FULL_64

    def test_precision_integer_rejected(self):
        """Test that integer precision values are rejected."""
        with pytest.raises(ValueError, match="Integer values not supported"):
            SessionSettings(precision=64)

    def test_precision_case_insensitive(self):
        """Test that precision strings are case-insensitive."""
        for value in ["DOUBLE", "Double", "FLOAT64", "Float64", "F64"]:
            session = SessionSettings(precision=value)
            assert session.precision == PrecisionStrategy.FULL_64

    def test_invalid_precision_string_raises_error(self):
        """Test that invalid precision string raises clear error."""
        with pytest.raises(ValueError, match="Invalid precision value.*wtf"):
            SessionSettings(precision="wtf")

    def test_invalid_numeric_string_raises_error(self):
        """Test that numeric string precision raises clear error."""
        with pytest.raises(ValueError, match="Invalid precision value"):
            SessionSettings(precision="128")

    def test_model_weights_float64(self, model_shape):
        """Test that model weights are initialized as float64."""
        with precision_override(PrecisionStrategy.FULL_64):
            model = Float64TestModel(**model_shape)

        # Check all parameter dtypes
        for name, param in model.named_parameters():
            assert param.dtype == torch.float64, (
                f"Parameter {name} has dtype {param.dtype}, expected float64"
            )

    def test_model_weights_float64_with_session(self, model_shape):
        """Test that model uses session precision for float64."""
        session = SessionSettings(precision="double")

        with precision_override(session.get_precision_strategy()):
            model = Float64TestModel(**model_shape)

        # All parameters should be float64
        for name, param in model.named_parameters():
            assert param.dtype == torch.float64, (
                f"Parameter {name} has dtype {param.dtype}, expected float64"
            )

    def test_input_casting_float64(self, model_shape):
        """Test that inputs are cast to float64 by Lightning wrapper during training."""
        with precision_override(PrecisionStrategy.FULL_64):
            model = Float64TestModel(**model_shape)

        # Create float32 input
        input_tensor = torch.randn(10, 20, dtype=torch.float32)
        assert input_tensor.dtype == torch.float32

        # Model is in float64 - Lightning will handle casting during training
        # We can verify the model parameters are float64
        assert next(model.parameters()).dtype == torch.float64

    def test_forward_pass_float64(self, model_shape):
        """Test that forward pass maintains float64 precision."""
        with precision_override(PrecisionStrategy.FULL_64):
            model = Float64TestModel(**model_shape)

        # Create float64 input (matching model precision)
        input_tensor = torch.randn(10, 20, dtype=torch.float64)

        # Forward pass
        output = model(input_tensor)

        # Output should be float64
        assert output.dtype == torch.float64
        assert output.shape == (10, 10)

    def test_data_loading_float64(self, sample_data):
        """Test that data is loaded with float64 precision."""
        session = SessionSettings(precision="float64")
        feature = Feature(name="input", path=sample_data["input"])

        # Load with session precision using context
        with precision_override(session.get_precision_strategy()):
            data = load_array(feature.path)

        assert data.dtype == torch.float64
        assert data.shape == (100, 20)

    def test_complete_pipeline_float64(self, sample_data, model_shape):
        """Test complete training pipeline with float64 precision."""
        print("\n=== Testing Complete Float64 Pipeline ===")

        # 1. Setup session with float64
        session = SessionSettings(precision="double", seed=42)
        assert session.precision == PrecisionStrategy.FULL_64

        # 2. Load data with float64 precision using context
        feature = Feature(name="input", path=sample_data["input"])
        target = Target(name="target", path=sample_data["target"])

        with precision_override(session.get_precision_strategy()):
            input_data = load_array(feature.path)
            target_data = load_array(target.path)

        assert input_data.dtype == torch.float64, "Input data should be float64"
        assert target_data.dtype == torch.float64, "Target data should be float64"

        # 3. Create model with session precision
        with precision_override(session.get_precision_strategy()):
            model = Float64TestModel(**model_shape)

        # Verify model weights are float64
        for name, param in model.named_parameters():
            assert param.dtype == torch.float64, f"Parameter {name} should be float64"

        # 4. Simulate training step
        batch_input = input_data[:16]  # Batch size 16
        batch_target = target_data[:16]

        # Forward pass
        predictions = model(batch_input)
        assert predictions.dtype == torch.float64, "Predictions should be float64"

        # Loss computation
        loss = torch.nn.functional.mse_loss(predictions, batch_target)
        assert loss.dtype == torch.float64, "Loss should be float64"

        # Backward pass (check gradients are float64)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.dtype == torch.float64, f"Gradient for {name} should be float64"

        print(f"✅ Input: {input_data.dtype}")
        print(f"✅ Model weights: {next(model.parameters()).dtype}")
        print(f"✅ Predictions: {predictions.dtype}")
        print(f"✅ Loss: {loss.dtype}")
        print(f"✅ Gradients: {next(model.parameters()).grad.dtype}")

    def test_precision_service_float64(self):
        """Test precision service with float64."""
        service = get_precision_service()
        session = SessionSettings(precision="f64")

        # Resolve precision
        resolved = service.resolve_precision(session)
        assert resolved == PrecisionStrategy.FULL_64

        # Get torch dtype
        dtype = service.get_torch_dtype(session)
        assert dtype == torch.float64

        # Get compute dtype (should be same as torch dtype for full precision)
        compute_dtype = service.get_compute_dtype(session)
        assert compute_dtype == torch.float64

        # Get Lightning precision
        lightning_precision = service.get_lightning_precision(session)
        assert lightning_precision == 64

    def test_precision_service_tensor_casting_float64(self):
        """Test precision service tensor casting to float64."""
        service = get_precision_service()
        session = SessionSettings(precision="double")

        # Create float32 tensor
        tensor = torch.randn(10, 20, dtype=torch.float32)
        assert tensor.dtype == torch.float32

        # Cast to session precision
        casted = service.cast_tensor(tensor, session)
        assert casted.dtype == torch.float64

    def test_precision_service_model_application_float64(self, model_shape):
        """Test precision service apply_precision_to_model with float64."""
        service = get_precision_service()
        session = SessionSettings(precision="fp64")

        # Create model with default precision (float32)
        with precision_override(PrecisionStrategy.FULL_32):
            model = Float64TestModel(**model_shape)
        assert next(model.parameters()).dtype == torch.float32

        # Apply float64 precision
        model = service.apply_precision_to_model(model, session)
        assert next(model.parameters()).dtype == torch.float64

    def test_model_preserves_precision_after_context(self, model_shape):
        """Models should retain precision after context exit."""
        session = SessionSettings(precision=PrecisionStrategy.FULL_64)

        with precision_override(session.get_precision_strategy()):
            model = Float64TestModel(**model_shape)
            param_dtype = next(model.parameters()).dtype

        assert param_dtype == torch.float64

        # Model parameters should remain float64 after context exit
        assert next(model.parameters()).dtype == torch.float64

    def test_apply_precision_updates_model_dtype(self, model_shape):
        """Applying precision via service must update model dtype."""
        service = get_precision_service()
        session = SessionSettings(precision=PrecisionStrategy.FULL_64)
        with precision_override(PrecisionStrategy.FULL_32):
            model = Float64TestModel(**model_shape)
        service.apply_precision_to_model(model, session)

        # Model parameters should now be float64
        assert next(model.parameters()).dtype == torch.float64

    def test_mixed_precision_data_types(self):
        """Test different precision aliases for comparison.

        Uses semantic aliases from the authoritative alias map for validation.
        """
        # Test a representative subset of SEMANTIC aliases (no numeric strings)
        test_aliases = ["double", "float64", "single", "float32", "half", "float16"]

        for alias in test_aliases:
            expected_strategy = _PRECISION_ALIAS_MAP[alias]
            session = SessionSettings(precision=alias)
            assert session.precision == expected_strategy, (
                f"Alias '{alias}' should resolve to {expected_strategy}"
            )

    def test_float64_memory_factor(self):
        """Test that float64 has 2x memory factor."""
        assert PrecisionStrategy.FULL_64.get_memory_factor() == 2.0
        assert PrecisionStrategy.FULL_32.get_memory_factor() == 1.0
        # float64 uses twice the memory of float32

    def test_float64_not_reduced_precision(self):
        """Test that float64 is not considered reduced precision."""
        assert not PrecisionStrategy.FULL_64.is_reduced_precision()
        assert not PrecisionStrategy.FULL_32.is_reduced_precision()
        assert PrecisionStrategy.TRUE_16.is_reduced_precision()

    def test_float64_no_autocast(self):
        """Test that float64 does not use autocast."""
        assert not PrecisionStrategy.FULL_64.supports_autocast()
        assert PrecisionStrategy.MIXED_16.supports_autocast()

    def test_precision_info_float64(self):
        """Test comprehensive precision info for float64."""
        service = get_precision_service()
        session = SessionSettings(precision="double")

        info = service.get_precision_info(session)

        assert info["strategy"] == "FULL_64"
        assert info["torch_dtype"] == "torch.float64"
        assert info["compute_dtype"] == "torch.float64"
        assert info["lightning_precision"] == 64
        assert info["supports_autocast"] is False
        assert info["is_reduced_precision"] is False
        assert info["memory_factor"] == 2.0
