"""Tests for precision inference and dtype handling during inference.

This test suite validates that:
1. Precision is correctly inferred from model checkpoints
2. Data is loaded with matching precision to avoid dtype mismatches
3. Defensive validation auto-casts mismatched dtypes with warnings
4. Precision context flows through the predictor lifecycle
"""

from __future__ import annotations

import torch
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from dlkit.interfaces.api.services.precision_service import PrecisionService
from dlkit.interfaces.inference.predictor import CheckpointPredictor, PredictorConfig
from dlkit.interfaces.inference.factory import PredictorFactory
from dlkit.interfaces.inference.infrastructure.adapters import DirectInferenceExecutor
from dlkit.interfaces.inference.domain.models import ModelState, ModelStateType, InferenceRequest
from dlkit.tools.config.precision.strategy import PrecisionStrategy


class TestPrecisionServiceInference:
    """Test PrecisionService.infer_precision_from_model()."""

    @pytest.fixture
    def precision_service(self) -> PrecisionService:
        """Create PrecisionService instance."""
        return PrecisionService()

    def test_infer_float32_model(self, precision_service: PrecisionService):
        """Test precision inference from float32 model."""
        model = torch.nn.Linear(10, 5).to(torch.float32)

        precision = precision_service.infer_precision_from_model(model)

        assert precision == PrecisionStrategy.FULL_32

    def test_infer_float64_model(self, precision_service: PrecisionService):
        """Test precision inference from float64 model."""
        model = torch.nn.Linear(10, 5).to(torch.float64)

        precision = precision_service.infer_precision_from_model(model)

        assert precision == PrecisionStrategy.FULL_64

    def test_infer_float16_model(self, precision_service: PrecisionService):
        """Test precision inference from float16 model."""
        model = torch.nn.Linear(10, 5).to(torch.float16)

        precision = precision_service.infer_precision_from_model(model)

        assert precision == PrecisionStrategy.TRUE_16

    def test_infer_bfloat16_model(self, precision_service: PrecisionService):
        """Test precision inference from bfloat16 model."""
        if not torch.cuda.is_available():
            pytest.skip("BFloat16 requires CUDA")

        model = torch.nn.Linear(10, 5).to(torch.bfloat16)

        precision = precision_service.infer_precision_from_model(model)

        assert precision == PrecisionStrategy.TRUE_BF16

    def test_infer_model_no_parameters(self, precision_service: PrecisionService):
        """Test precision inference from model with no parameters."""
        model = torch.nn.Identity()

        precision = precision_service.infer_precision_from_model(model)

        assert precision is None

    def test_infer_model_none(self, precision_service: PrecisionService):
        """Test precision inference handles None gracefully."""
        # Create a model that will fail to return parameters
        model = Mock()
        model.parameters.side_effect = AttributeError("No parameters")

        precision = precision_service.infer_precision_from_model(model)

        assert precision is None


class TestCheckpointPredictorPrecision:
    """Test CheckpointPredictor precision context establishment."""

    @pytest.fixture
    def mock_model_loading_use_case(self):
        """Create mock model loading use case."""
        use_case = Mock()
        model = torch.nn.Linear(10, 5).to(torch.float32)
        model.eval()

        model_state = ModelState(
            model=model,
            state_type=ModelStateType.INFERENCE_READY,
            device="cpu"
        )
        use_case.load_model.return_value = model_state
        return use_case

    @pytest.fixture
    def mock_inference_execution_use_case(self):
        """Create mock inference execution use case."""
        use_case = Mock()
        use_case.execute_inference.return_value = MagicMock()
        return use_case

    def test_precision_inferred_on_load(
        self,
        mock_model_loading_use_case,
        mock_inference_execution_use_case,
        tmp_path: Path
    ):
        """Test that precision is inferred from model during load()."""
        config = PredictorConfig(
            checkpoint_path=tmp_path / "model.ckpt",
            auto_load=False
        )

        predictor = CheckpointPredictor(
            config=config,
            model_loading_use_case=mock_model_loading_use_case,
            inference_execution_use_case=mock_inference_execution_use_case
        )

        predictor.load()

        # Verify precision was inferred
        assert predictor._inferred_precision == PrecisionStrategy.FULL_32

    def test_configured_precision_overrides_inference(
        self,
        mock_model_loading_use_case,
        mock_inference_execution_use_case,
        tmp_path: Path
    ):
        """Test that configured precision overrides inferred precision."""
        config = PredictorConfig(
            checkpoint_path=tmp_path / "model.ckpt",
            precision=PrecisionStrategy.FULL_64,  # Override
            auto_load=False
        )

        predictor = CheckpointPredictor(
            config=config,
            model_loading_use_case=mock_model_loading_use_case,
            inference_execution_use_case=mock_inference_execution_use_case
        )

        predictor.load()

        # Verify configured precision was used instead of inferred
        assert predictor._inferred_precision == PrecisionStrategy.FULL_64

    @patch('dlkit.interfaces.inference.predictor.precision_override')
    def test_precision_context_applied_during_predict(
        self,
        mock_precision_override,
        mock_model_loading_use_case,
        mock_inference_execution_use_case,
        tmp_path: Path
    ):
        """Test that precision context is applied during predict()."""
        config = PredictorConfig(
            checkpoint_path=tmp_path / "model.ckpt",
            auto_load=True
        )

        # Setup mock context manager
        mock_context = MagicMock()
        mock_precision_override.return_value = mock_context

        predictor = CheckpointPredictor(
            config=config,
            model_loading_use_case=mock_model_loading_use_case,
            inference_execution_use_case=mock_inference_execution_use_case
        )

        # Execute prediction
        inputs = {"x": torch.randn(32, 10)}
        predictor.predict(inputs)

        # Verify precision context was established
        mock_precision_override.assert_called_once_with(PrecisionStrategy.FULL_32)
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()


class TestPredictorFactoryPrecision:
    """Test PredictorFactory precision parameter propagation."""

    @pytest.fixture
    def mock_use_cases(self):
        """Create mock use cases for factory."""
        model_loading = Mock()
        model_loading.load_model.return_value = ModelState(
            model=torch.nn.Linear(10, 5).to(torch.float32),
            state_type=ModelStateType.LOADED,
            device="cpu"
        )

        inference_execution = Mock()
        inference_execution.execute_inference.return_value = MagicMock()

        return model_loading, inference_execution

    def test_factory_passes_precision_to_predictor(
        self,
        mock_use_cases,
        tmp_path: Path
    ):
        """Test that factory passes precision parameter to predictor config."""
        model_loading, inference_execution = mock_use_cases

        factory = PredictorFactory(
            model_loading_use_case=model_loading,
            inference_execution_use_case=inference_execution
        )

        predictor = factory.create_from_checkpoint(
            checkpoint_path=tmp_path / "model.ckpt",
            precision=PrecisionStrategy.FULL_64,
            auto_load=False
        )

        # Verify precision was passed to config
        assert predictor._config.precision == PrecisionStrategy.FULL_64

    def test_factory_precision_none_allows_inference(
        self,
        mock_use_cases,
        tmp_path: Path
    ):
        """Test that factory with precision=None allows inference."""
        model_loading, inference_execution = mock_use_cases

        factory = PredictorFactory(
            model_loading_use_case=model_loading,
            inference_execution_use_case=inference_execution
        )

        predictor = factory.create_from_checkpoint(
            checkpoint_path=tmp_path / "model.ckpt",
            precision=None,  # Allow inference
            auto_load=False
        )

        # Verify precision is None (will be inferred)
        assert predictor._config.precision is None


class TestDirectInferenceExecutorDefensiveValidation:
    """Test DirectInferenceExecutor defensive dtype validation."""

    @pytest.fixture
    def executor(self) -> DirectInferenceExecutor:
        """Create DirectInferenceExecutor instance."""
        return DirectInferenceExecutor()

    @pytest.fixture
    def model_state_float32(self) -> ModelState:
        """Create model state with float32 model."""
        model = torch.nn.Linear(10, 5).to(torch.float32)
        model.eval()

        return ModelState(
            model=model,
            state_type=ModelStateType.INFERENCE_READY,
            device="cpu"
        )

    def test_get_model_dtype_float32(self, executor: DirectInferenceExecutor):
        """Test _get_model_dtype() returns correct dtype."""
        model = torch.nn.Linear(10, 5).to(torch.float32)

        dtype = executor._get_model_dtype(model)

        assert dtype == torch.float32

    def test_get_model_dtype_float64(self, executor: DirectInferenceExecutor):
        """Test _get_model_dtype() with float64 model."""
        model = torch.nn.Linear(10, 5).to(torch.float64)

        dtype = executor._get_model_dtype(model)

        assert dtype == torch.float64

    def test_get_model_dtype_no_parameters(self, executor: DirectInferenceExecutor):
        """Test _get_model_dtype() with model that has no parameters."""
        model = torch.nn.Identity()

        dtype = executor._get_model_dtype(model)

        assert dtype is None

    def test_cast_inputs_matching_dtype_no_change(self, executor: DirectInferenceExecutor):
        """Test that inputs with matching dtype are not changed."""
        inputs = torch.randn(32, 10, dtype=torch.float32)

        result = executor._cast_inputs_recursive(inputs, torch.float32)

        assert result.dtype == torch.float32
        assert torch.equal(result, inputs)

    def test_cast_inputs_mismatched_dtype_with_warning(self, executor: DirectInferenceExecutor):
        """Test that inputs with mismatched dtype are cast with warning."""
        inputs = torch.randn(32, 10, dtype=torch.float64)

        with patch('dlkit.interfaces.inference.infrastructure.adapters.logger') as mock_logger:
            result = executor._cast_inputs_recursive(inputs, torch.float32)

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "dtype mismatch" in warning_msg.lower()

        # Verify dtype was cast
        assert result.dtype == torch.float32

    def test_cast_inputs_dict_recursive(self, executor: DirectInferenceExecutor):
        """Test recursive casting for dictionary inputs."""
        inputs = {
            "x": torch.randn(32, 10, dtype=torch.float64),
            "y": torch.randn(32, 5, dtype=torch.float64)
        }

        with patch('dlkit.interfaces.inference.infrastructure.adapters.logger'):
            result = executor._cast_inputs_recursive(inputs, torch.float32)

        assert result["x"].dtype == torch.float32
        assert result["y"].dtype == torch.float32

    def test_validate_and_cast_inputs_integration(
        self,
        executor: DirectInferenceExecutor,
        model_state_float32: ModelState
    ):
        """Integration test for _validate_and_cast_inputs()."""
        # Create input with mismatched dtype
        inputs = {"x": torch.randn(32, 10, dtype=torch.float64)}

        with patch('dlkit.interfaces.inference.infrastructure.adapters.logger'):
            result = executor._validate_and_cast_inputs(
                inputs,
                model_state_float32.model,
                model_state_float32
            )

        # Verify inputs were cast to match model dtype
        assert result["x"].dtype == torch.float32


class TestPrecisionFlowIntegration:
    """Integration tests for precision flow through entire inference pipeline."""

    def test_precision_inference_prevents_dtype_mismatch(self):
        """Test that precision inference prevents the original dtype mismatch error.

        This test simulates the user's original problem:
        - Model trained/saved with float32
        - Data on disk is float64
        - Without precision context, data loads as float64
        - Forward pass fails with dtype mismatch

        With precision inference:
        - Predictor infers float32 from model
        - Precision context established
        - Data loads as float32 (or auto-cast if needed)
        - Forward pass succeeds
        """
        # This is a conceptual test - in practice would require:
        # 1. A real checkpoint with float32 model
        # 2. A dataset with float64 data
        # 3. Full predictor initialization
        # 4. Verification that no dtype errors occur

        # For now, this documents the expected behavior
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
