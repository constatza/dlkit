"""Integration test for dtype mismatch fix.

This test reproduces the EXACT scenario from the user's bug report:
- Model trained/saved with float32
- Numpy array input with float64
- Without fix: dtype mismatch error
- With fix: automatic conversion to float32
"""

import torch
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from dlkit.interfaces.inference.predictor import CheckpointPredictor, PredictorConfig
from dlkit.interfaces.inference.domain.models import ModelState, ModelStateType
from dlkit.tools.config.precision.strategy import PrecisionStrategy
from dlkit.tools.config import GeneralSettings


def test_dtype_mismatch_real_scenario(tmp_path: Path):
    """Test that reproduces the exact user scenario and verifies the fix.

    Scenario:
    - Model has float32 weights (typical for PyTorch models)
    - User passes numpy array with float64 dtype (numpy default)
    - Should automatically convert to float32 without error
    """
    # Create a float32 model (typical PyTorch model)
    model = torch.nn.Linear(504, 504).to(torch.float32)
    model.eval()

    # Mock the model loading use case to return our float32 model
    mock_model_loading = Mock()
    model_state = ModelState(
        model=model,
        state_type=ModelStateType.INFERENCE_READY,
        device="cpu"
    )
    mock_model_loading.load_model.return_value = model_state

    # Mock inference execution to just run the model
    mock_inference_execution = Mock()

    def execute_inference_side_effect(model_state, request):
        """Execute actual model forward pass."""
        inputs = request.inputs
        # Get the input tensor
        input_tensor = inputs['rhs']

        # This is where the dtype mismatch would occur
        # Model is float32, input might be float64
        output = model_state.model(input_tensor)

        return {"predictions": output}

    mock_inference_execution.execute_inference.side_effect = execute_inference_side_effect

    # Create predictor
    config = PredictorConfig(
        checkpoint_path=tmp_path / "model.ckpt",
        auto_load=True
    )

    predictor = CheckpointPredictor(
        config=config,
        model_loading_use_case=mock_model_loading,
        inference_execution_use_case=mock_inference_execution
    )

    # Create float64 numpy input (this is what numpy creates by default)
    batch = {
        'rhs': np.random.randn(32, 504).astype(np.float64)  # Explicitly float64
    }

    # Verify input is float64
    assert batch['rhs'].dtype == np.float64

    # THIS SHOULD NOT RAISE AN ERROR
    # The fix should:
    # 1. Infer float32 from model
    # 2. Convert numpy float64 to torch float32 during to_tensor_dict()
    result = predictor.predict(batch)

    # Verify the prediction worked
    assert result is not None

    # Verify precision was inferred correctly
    assert predictor._inferred_precision == PrecisionStrategy.FULL_32


def test_precision_inference_flow_end_to_end(tmp_path: Path):
    """End-to-end test of precision inference preventing dtype errors."""
    # Create model with specific dtype
    model = torch.nn.Linear(10, 5).to(torch.float32)
    model.eval()

    # Setup mocks
    mock_model_loading = Mock()
    model_state = ModelState(
        model=model,
        state_type=ModelStateType.INFERENCE_READY,
        device="cpu"
    )
    mock_model_loading.load_model.return_value = model_state

    mock_inference_execution = Mock()
    mock_inference_execution.execute_inference.return_value = {
        "predictions": torch.randn(32, 5)
    }

    # Create predictor
    config = PredictorConfig(
        checkpoint_path=tmp_path / "model.ckpt",
        auto_load=True
    )

    predictor = CheckpointPredictor(
        config=config,
        model_loading_use_case=mock_model_loading,
        inference_execution_use_case=mock_inference_execution
    )

    # Test with float64 numpy array
    inputs = {"x": np.random.randn(32, 10).astype(np.float64)}

    result = predictor.predict(inputs)

    # Verify call was made
    assert mock_inference_execution.execute_inference.called

    # Get the actual request that was passed (as keyword argument)
    call_kwargs = mock_inference_execution.execute_inference.call_args.kwargs
    actual_request = call_kwargs['request']

    # Verify the input tensor was converted to float32
    actual_input_tensor = actual_request.inputs['x']
    assert actual_input_tensor.dtype == torch.float32, \
        f"Expected float32 but got {actual_input_tensor.dtype}"


def test_predict_from_config_dtype_conversion(tmp_path: Path):
    """Test that predict_from_config() properly converts dtypes from dataloader.

    This test covers the EXACT bug that was missed:
    - predict_from_config() was not passing dtype parameter to to_tensor_dict()
    - This caused data from dataloader to remain in original dtype (float64)
    - Model expecting float32 would then get dtype mismatch error

    This is a regression test to ensure the bug never comes back.
    """
    # Create a float64 model (simulating user's scenario)
    model = torch.nn.Linear(10, 5).to(torch.float64)
    model.eval()

    # Mock the model loading use case to return our float64 model
    mock_model_loading = Mock()
    model_state = ModelState(
        model=model,
        state_type=ModelStateType.INFERENCE_READY,
        device="cpu"
    )
    mock_model_loading.load_model.return_value = model_state

    # Track what dtypes are passed to inference execution
    received_dtypes = []

    def execute_inference_side_effect(model_state, request):
        """Execute inference and track the input dtypes."""
        inputs = request.inputs
        # Record the dtype we received
        for key, tensor in inputs.items():
            if torch.is_tensor(tensor):
                received_dtypes.append(tensor.dtype)

        # Run the model forward pass (would fail if dtype mismatch)
        input_tensor = next(iter(inputs.values()))
        output = model_state.model(input_tensor)

        return {"predictions": output}

    mock_inference_execution = Mock()
    mock_inference_execution.execute_inference.side_effect = execute_inference_side_effect

    # Create predictor
    config = PredictorConfig(
        checkpoint_path=tmp_path / "model.ckpt",
        auto_load=True
    )

    predictor = CheckpointPredictor(
        config=config,
        model_loading_use_case=mock_model_loading,
        inference_execution_use_case=mock_inference_execution
    )

    # Verify precision was inferred as float64
    assert predictor._inferred_precision == PrecisionStrategy.FULL_64

    # Create a mock dataloader that returns batches with float64 numpy arrays
    # (simulating what happens when data is loaded from disk)
    batch1 = {"features": np.random.randn(8, 10).astype(np.float64)}
    batch2 = {"features": np.random.randn(8, 10).astype(np.float64)}
    mock_dataloader = iter([batch1, batch2])

    # Mock the config and dataloader creation
    mock_config = Mock(spec=GeneralSettings)

    with patch("dlkit.tools.config.load_settings") as mock_load_settings, \
         patch("dlkit.interfaces.inference.application.config_inference.build_prediction_dataloader_from_config") as mock_build_dataloader:

        mock_load_settings.return_value = mock_config
        mock_build_dataloader.return_value = mock_dataloader

        # Call predict_from_config - this is the method that had the bug!
        results = list(predictor.predict_from_config(tmp_path / "config.toml"))

        # Verify we got results
        assert len(results) == 2

        # CRITICAL ASSERTION: Verify all inputs were converted to float64 (model's dtype)
        # Without the fix, they would have been float32 (default) causing dtype mismatch
        assert len(received_dtypes) == 2
        assert all(dtype == torch.float64 for dtype in received_dtypes), \
            f"Expected all inputs to be float64, but got: {received_dtypes}"

        # Verify the dataloader builder was called with correct precision context
        # (the precision context should have been set before building dataloader)
        mock_build_dataloader.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
