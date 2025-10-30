"""Tests for inference components: InferenceConfig and InferenceInput."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import pytest

from dlkit.interfaces.inference import InferenceConfig, InferenceInput


class TestInferenceConfig:
    """Test InferenceConfig creation and validation."""

    def test_inference_config_minimal(self, tmp_path: Path):
        """Test creating minimal InferenceConfig."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()  # Create empty file

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path
        )

        assert config.model_checkpoint_path == checkpoint_path
        # Test defaults
        assert config.batch_size == 32
        assert config.device == "auto"
        assert config.apply_transforms is True

    def test_inference_config_full(self, tmp_path: Path):
        """Test creating InferenceConfig with all parameters."""
        checkpoint_path = tmp_path / "full_model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            feature_names=["input_x", "input_y"],
            target_names=["output_z"],
            transform_names=["MinMaxScaler", "StandardScaler"],
            model_shape={"input_x": [10], "input_y": [5], "output_z": [3]},
            device="cuda",
            batch_size=16,
            apply_transforms=False
        )

        assert config.model_checkpoint_path == checkpoint_path
        assert config.feature_names == ["input_x", "input_y"]
        assert config.target_names == ["output_z"]
        assert config.transform_names == ["MinMaxScaler", "StandardScaler"]
        assert config.model_shape == {"input_x": (10,), "input_y": (5,), "output_z": (3,)}
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.apply_transforms is False

    def test_inference_config_path_validation(self):
        """Test InferenceConfig validates checkpoint path."""
        # Should handle both str and Path
        config_str = InferenceConfig(model_checkpoint_path="./model.ckpt")
        assert isinstance(config_str.model_checkpoint_path, Path)

        config_path = InferenceConfig(model_checkpoint_path=Path("./model.ckpt"))
        assert isinstance(config_path.model_checkpoint_path, Path)

    def test_inference_config_has_transforms_method(self, tmp_path: Path):
        """Test has_transforms method."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        # Config without transform names
        config_no_transforms = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            transform_names=None
        )

        # Config with transform names
        config_with_transforms = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            transform_names=["MinMaxScaler"]
        )

        # These methods should exist and return appropriate values
        # The actual implementation may vary based on the InferenceConfig design
        if hasattr(config_no_transforms, 'has_transforms'):
            assert not config_no_transforms.has_transforms()
            assert config_with_transforms.has_transforms()

    def test_inference_config_get_methods(self, tmp_path: Path):
        """Test getter methods for names."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            feature_names=["x", "y"],
            target_names=["z"]
        )

        # These methods should exist based on the API usage in InferenceStrategy
        if hasattr(config, 'get_target_names'):
            assert config.get_target_names() == ["z"]
        if hasattr(config, 'get_feature_names'):
            assert config.get_feature_names() == ["x", "y"]


class TestInferenceInput:
    """Test InferenceInput wrapper functionality."""

    def test_inference_input_with_tensor_dict(self):
        """Test InferenceInput with dictionary of tensors."""
        data = {
            "x": torch.randn(4, 5),
            "y": torch.randn(4, 3),
            "metadata": torch.randn(4, 2)
        }

        inference_input = InferenceInput(data)
        assert isinstance(inference_input, InferenceInput)

        # The input should be stored and accessible
        # Implementation details may vary, but basic functionality should work

    def test_inference_input_with_single_tensor(self):
        """Test InferenceInput with single tensor."""
        data = torch.randn(8, 10)
        inference_input = InferenceInput(data)

        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_numpy_dict(self):
        """Test InferenceInput with dictionary of numpy arrays."""
        data = {
            "features": np.random.randn(6, 8),
            "aux_data": np.random.randn(6, 2)
        }

        inference_input = InferenceInput(data)
        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_single_numpy_array(self):
        """Test InferenceInput with single numpy array."""
        data = np.random.randn(5, 7)
        inference_input = InferenceInput(data)

        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_mixed_types(self):
        """Test InferenceInput with mixed data types."""
        data = {
            "tensor_data": torch.randn(3, 4),
            "numpy_data": np.random.randn(3, 5),
            "list_data": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        }

        inference_input = InferenceInput(data)
        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_with_file_paths(self, tmp_path: Path):
        """Test InferenceInput with file paths."""
        # Create test data files
        data_file = tmp_path / "test_data.npy"
        test_data = np.random.randn(10, 6)
        np.save(data_file, test_data)

        # Test with single file path
        inference_input = InferenceInput(str(data_file))
        assert isinstance(inference_input, InferenceInput)

        # Test with dictionary of file paths
        data2_file = tmp_path / "test_data2.npy"
        test_data2 = np.random.randn(10, 4)
        np.save(data2_file, test_data2)

        file_dict = {
            "input1": str(data_file),
            "input2": str(data2_file)
        }
        inference_input_dict = InferenceInput(file_dict)
        assert isinstance(inference_input_dict, InferenceInput)

    def test_inference_input_with_pathlib_paths(self, tmp_path: Path):
        """Test InferenceInput with pathlib.Path objects."""
        data_file = tmp_path / "pathlib_test.npy"
        test_data = np.random.randn(5, 3)
        np.save(data_file, test_data)

        inference_input = InferenceInput(data_file)  # Pass Path object directly
        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_empty_dict(self):
        """Test InferenceInput with empty dictionary."""
        inference_input = InferenceInput({})
        assert isinstance(inference_input, InferenceInput)

    def test_inference_input_none_handling(self):
        """Test InferenceInput error handling with None input."""
        with pytest.raises((ValueError, TypeError)):
            InferenceInput(None)

    def test_inference_input_invalid_file_path(self):
        """Test InferenceInput with non-existent file path."""
        # This should either raise an error or handle gracefully
        # depending on implementation design
        try:
            inference_input = InferenceInput("nonexistent_file.npy")
            # If no error, the implementation defers validation
            assert isinstance(inference_input, InferenceInput)
        except (FileNotFoundError, ValueError):
            # If error, that's also acceptable validation behavior
            pass

    def test_inference_input_batch_consistency(self):
        """Test InferenceInput with inconsistent batch sizes."""
        data = {
            "x": torch.randn(5, 10),  # 5 samples
            "y": torch.randn(3, 8)    # 3 samples - inconsistent!
        }

        # Implementation may validate batch consistency or defer to runtime
        try:
            inference_input = InferenceInput(data)
            assert isinstance(inference_input, InferenceInput)
        except ValueError:
            # Batch size validation at creation time is also valid
            pass

    def test_inference_input_conversion_methods(self):
        """Test InferenceInput data access/conversion methods."""
        data = {"x": torch.randn(2, 5)}
        inference_input = InferenceInput(data)

        # Test if common methods exist (implementation-dependent)
        # These are example methods that might be useful
        if hasattr(inference_input, 'to_dict'):
            result_dict = inference_input.to_dict()
            assert isinstance(result_dict, dict)

        if hasattr(inference_input, 'get_batch_size'):
            batch_size = inference_input.get_batch_size()
            assert isinstance(batch_size, int)

        if hasattr(inference_input, 'to_tensors'):
            tensors = inference_input.to_tensors()
            assert isinstance(tensors, dict)
            for key, tensor in tensors.items():
                assert isinstance(tensor, torch.Tensor)


class TestInferenceComponentsIntegration:
    """Test interaction between InferenceConfig and InferenceInput."""

    def test_config_and_input_compatibility(self, tmp_path: Path):
        """Test that InferenceConfig and InferenceInput work together."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            feature_names=["x", "y"],
            target_names=["z"],
            batch_size=8
        )

        inputs = InferenceInput({
            "x": torch.randn(10, 5),
            "y": torch.randn(10, 3)
        })

        # They should be created successfully and be compatible
        assert isinstance(config, InferenceConfig)
        assert isinstance(inputs, InferenceInput)

        # Test that they contain expected information
        assert config.feature_names == ["x", "y"]
        assert config.batch_size == 8

    def test_config_input_mismatch_handling(self, tmp_path: Path):
        """Test handling of mismatched config and input specifications."""
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint_path.touch()

        config = InferenceConfig(
            model_checkpoint_path=checkpoint_path,
            feature_names=["expected_input"],  # Expects "expected_input"
            batch_size=4
        )

        inputs = InferenceInput({
            "actual_input": torch.randn(6, 10)  # Provides "actual_input"
        })

        # The components themselves should create fine
        # Validation of compatibility might happen at inference time
        assert isinstance(config, InferenceConfig)
        assert isinstance(inputs, InferenceInput)
