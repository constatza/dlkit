"""Integration tests for the new inference API methods."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np
import pytest

from dlkit.interfaces.api import infer, predict_with_config
from dlkit.interfaces.inference import InferenceConfig, InferenceInput, InferenceService
from dlkit.interfaces.inference.strategies.inference_strategy import InferenceStrategy
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.tools.config import GeneralSettings


class TestDirectInferenceAPI:
    """Test the new direct inference API: infer(checkpoint_path, inputs)."""

    @pytest.fixture
    def mock_checkpoint_path(self, tmp_path):
        """Create a mock checkpoint file."""
        checkpoint_path = tmp_path / "test_model.ckpt"

        # Create a minimal checkpoint with required structure
        checkpoint_data = {
            "state_dict": {
                "linear.weight": torch.randn(10, 5),
                "linear.bias": torch.randn(10),
            },
            "dlkit_metadata": {
                "version": "2.0",
                "model_family": "dlkit_nn",
                "shape_spec": {
                    "data": {"x": [5], "y": [10]},
                    "model_family": "dlkit_nn",
                    "inferred_from": "training_dataset"
                },
                "model_settings": {
                    "name": "TestModel",
                    "module_path": "test.module",
                    "params": {"input_size": 5, "output_size": 10},
                    "class_name": "ModelComponentSettings"
                },
                "entry_configs": {}
            },
            "hyper_parameters": {},
            "epoch": 10,
            "global_step": 100,
        }

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def sample_tensor_inputs(self):
        """Sample tensor inputs for testing."""
        return {"x": torch.randn(4, 5)}

    @pytest.fixture
    def sample_numpy_inputs(self):
        """Sample numpy inputs for testing."""
        return {"x": np.random.randn(4, 5)}

    @patch('dlkit.interfaces.inference.container.get_inference_orchestrator')
    def test_infer_with_tensor_inputs(self, mock_get_orchestrator, mock_checkpoint_path, sample_tensor_inputs):
        """Test direct inference with tensor inputs."""
        # Mock the orchestrator and its response
        mock_orchestrator = Mock()
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(4, 10)},
            metrics=None,
            duration_seconds=1.5
        )
        mock_orchestrator.infer_from_checkpoint.return_value = mock_result
        mock_get_orchestrator.return_value = mock_orchestrator

        # Call the API
        result = infer(
            checkpoint_path=mock_checkpoint_path,
            inputs=sample_tensor_inputs,
            batch_size=16
        )

        # Verify
        assert isinstance(result, InferenceResult)
        assert "y" in result.predictions
        mock_orchestrator.infer_from_checkpoint.assert_called_once()

        # Check the call arguments
        call_args = mock_orchestrator.infer_from_checkpoint.call_args[1]
        assert call_args["checkpoint_path"] == mock_checkpoint_path
        assert call_args["inputs"] == sample_tensor_inputs
        assert call_args["batch_size"] == 16
        assert call_args["device"] == "auto"
        assert call_args["apply_transforms"] == True

    @patch('dlkit.interfaces.inference.container.get_inference_orchestrator')
    def test_infer_with_numpy_inputs(self, mock_get_orchestrator, mock_checkpoint_path, sample_numpy_inputs):
        """Test direct inference with numpy inputs."""
        mock_orchestrator = Mock()
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(4, 10)},
            metrics=None,
            duration_seconds=1.2
        )
        mock_orchestrator.infer_from_checkpoint.return_value = mock_result
        mock_get_orchestrator.return_value = mock_orchestrator

        result = infer(
            checkpoint_path=mock_checkpoint_path,
            inputs=sample_numpy_inputs,
            device="cpu",
            apply_transforms=False
        )

        assert isinstance(result, InferenceResult)
        mock_orchestrator.infer_from_checkpoint.assert_called_once()

        call_args = mock_orchestrator.infer_from_checkpoint.call_args[1]
        assert call_args["device"] == "cpu"
        assert call_args["apply_transforms"] is False

    @patch('dlkit.interfaces.inference.container.get_inference_orchestrator')
    def test_infer_with_file_path_inputs(self, mock_get_orchestrator, mock_checkpoint_path, tmp_path):
        """Test direct inference with file path inputs."""
        # Create test data file
        data_file = tmp_path / "test_data.npy"
        test_data = np.random.randn(4, 5)
        np.save(data_file, test_data)

        mock_orchestrator = Mock()
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(4, 10)},
            metrics=None,
            duration_seconds=0.8
        )
        mock_orchestrator.infer_from_checkpoint.return_value = mock_result
        mock_get_orchestrator.return_value = mock_orchestrator

        result = infer(
            checkpoint_path=mock_checkpoint_path,
            inputs=str(data_file)
        )

        assert isinstance(result, InferenceResult)
        mock_orchestrator.infer_from_checkpoint.assert_called_once()

    @patch('dlkit.interfaces.inference.container.get_inference_orchestrator')
    def test_infer_with_inference_input_wrapper(self, mock_get_orchestrator, mock_checkpoint_path, sample_tensor_inputs):
        """Test direct inference with pre-wrapped InferenceInput."""
        inference_input = InferenceInput(sample_tensor_inputs)

        mock_orchestrator = Mock()
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(4, 10)},
            metrics=None,
            duration_seconds=1.1
        )
        mock_orchestrator.infer_from_checkpoint.return_value = mock_result
        mock_get_orchestrator.return_value = mock_orchestrator

        result = infer(
            checkpoint_path=mock_checkpoint_path,
            inputs=inference_input
        )

        assert isinstance(result, InferenceResult)
        mock_orchestrator.infer_from_checkpoint.assert_called_once()

        # Check the input was passed correctly
        call_args = mock_orchestrator.infer_from_checkpoint.call_args[1]
        assert call_args["inputs"] is inference_input


class TestPredictWithConfigAPI:
    """Test the predict_with_config API for Lightning-based inference."""

    @pytest.fixture
    def mock_training_settings(self):
        """Create mock training settings."""
        return GeneralSettings()

    @pytest.fixture
    def mock_checkpoint_path(self, tmp_path):
        """Create a mock checkpoint file."""
        checkpoint_path = tmp_path / "test_model.ckpt"
        checkpoint_path.touch()  # Create empty file for now
        return checkpoint_path

    @patch.object(InferenceService, 'predict')
    def test_predict_with_config_basic(self, mock_predict, mock_training_settings, mock_checkpoint_path):
        """Test basic predict_with_config functionality."""
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(10, 5)},
            metrics={"accuracy": 0.85},
            duration_seconds=2.1
        )
        mock_predict.return_value = mock_result

        result = predict_with_config(
            training_settings=mock_training_settings,
            checkpoint_path=mock_checkpoint_path
        )

        assert isinstance(result, InferenceResult)
        assert "y" in result.predictions
        assert result.metrics["accuracy"] == 0.85
        mock_predict.assert_called_once()

    @patch.object(InferenceService, 'predict')
    def test_predict_with_config_overrides(self, mock_predict, mock_training_settings, mock_checkpoint_path):
        """Test predict_with_config with parameter overrides."""
        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(20, 5)},
            metrics=None,
            duration_seconds=1.8
        )
        mock_predict.return_value = mock_result

        result = predict_with_config(
            training_settings=mock_training_settings,
            checkpoint_path=mock_checkpoint_path,
            batch_size=32,
            device="cuda",
            custom_param="test_value"
        )

        assert isinstance(result, InferenceResult)
        mock_predict.assert_called_once()

        # Verify overrides were passed
        call_args = mock_predict.call_args
        assert call_args[1]["batch_size"] == 32
        assert call_args[1]["device"] == "cuda"
        assert call_args[1]["custom_param"] == "test_value"


class TestInferenceConfig:
    """Test InferenceConfig creation and usage."""

    def test_inference_config_creation(self, tmp_path):
        """Test creating InferenceConfig manually."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            batch_size=16,
            device="cuda",
            apply_transforms=True
        )

        assert config.model_checkpoint_path == checkpoint_path
        assert config.batch_size == 16
        assert config.device == "cuda"
        assert config.apply_transforms is True

    @patch.object(InferenceService, 'infer_with_config')
    def test_infer_with_config_api(self, mock_infer_with_config, tmp_path):
        """Test using InferenceConfig with the infer_with_config API."""
        from dlkit.interfaces.inference.api import infer_with_config

        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            batch_size=8,
            device="cpu"
        )

        inputs = {"x": torch.randn(2, 10)}

        mock_result = InferenceResult(
            model_state=Mock(),
            predictions={"y": torch.randn(2, 5)},
            metrics=None,
            duration_seconds=0.5
        )
        mock_infer_with_config.return_value = mock_result

        result = infer_with_config(config, inputs, batch_size=16)

        assert isinstance(result, InferenceResult)
        mock_infer_with_config.assert_called_once()


class TestInferenceInput:
    """Test InferenceInput wrapper functionality."""

    def test_inference_input_with_dict(self):
        """Test InferenceInput with dictionary input."""
        data = {"x": torch.randn(4, 5), "metadata": torch.randn(4, 2)}
        inference_input = InferenceInput(data)

        assert isinstance(inference_input, InferenceInput)
        # Add more specific tests once we know the InferenceInput implementation

    def test_inference_input_with_single_tensor(self):
        """Test InferenceInput with single tensor."""
        data = torch.randn(4, 10)
        inference_input = InferenceInput(data)

        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_numpy_array(self):
        """Test InferenceInput with numpy array."""
        data = np.random.randn(4, 8)
        inference_input = InferenceInput(data)

        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_file_path(self, tmp_path):
        """Test InferenceInput with file path."""
        data_file = tmp_path / "data.npy"
        np.save(data_file, np.random.randn(3, 6))

        inference_input = InferenceInput(str(data_file))

        assert isinstance(inference_input, InferenceInput)


class TestInferenceAPIErrorHandling:
    """Test error handling in inference APIs."""

    def test_infer_with_invalid_checkpoint(self):
        """Test infer() with non-existent checkpoint."""
        from dlkit.interfaces.api.domain.errors import WorkflowError
        with pytest.raises(WorkflowError, match="Checkpoint not found"):
            infer(
                checkpoint_path="nonexistent_checkpoint.ckpt",
                inputs={"x": torch.randn(2, 5)}
            )

    def test_infer_with_none_inputs(self, tmp_path):
        """Test infer() with None inputs."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        with pytest.raises((ValueError, TypeError)):
            infer(
                checkpoint_path=checkpoint_path,
                inputs=None
            )

    def test_predict_with_config_invalid_settings(self, tmp_path):
        """Test predict_with_config with invalid settings."""
        from dlkit.interfaces.api.domain.errors import WorkflowError
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        with pytest.raises(WorkflowError, match="Simple prediction failed"):
            predict_with_config(
                training_settings=None,  # Invalid
                checkpoint_path=checkpoint_path
            )


class TestInferenceAPIImports:
    """Test that all inference APIs can be imported correctly."""

    def test_import_main_api_functions(self):
        """Test importing main API functions."""
        from dlkit.interfaces.api import infer, predict_with_config

        assert callable(infer)
        assert callable(predict_with_config)

    def test_import_inference_components(self):
        """Test importing inference components."""
        from dlkit.interfaces.inference import InferenceConfig, InferenceInput, InferenceService

        assert InferenceConfig is not None
        assert InferenceInput is not None
        assert InferenceService is not None

    def test_import_inference_strategy(self):
        """Test importing inference strategy."""
        from dlkit.interfaces.inference.strategies.inference_strategy import InferenceStrategy

        assert InferenceStrategy is not None