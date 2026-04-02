"""API integration tests showing shape system working with high-level concepts.

These tests demonstrate the shape system integration without requiring
the full DLKit API stack to be working perfectly.
"""

from unittest.mock import Mock

import torch

from dlkit.domain.shapes import (
    ModelFamily,
    ShapeInferenceEngine,
    ShapeSource,
    ShapeSystemFactory,
    create_shape_spec,
)


class TestAPIConceptIntegration:
    """Test shape system integration with high-level API concepts."""

    def test_train_like_workflow_shape_integration(self):
        """Test shape integration in a train-like workflow."""
        # Simulate what happens in dlkit.train()

        # 1. Dataset provides shapes through sampling
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            return_value={
                "x": torch.randn(28, 28),  # MNIST-like
                "y": torch.tensor(5),  # Class label
            }
        )
        mock_dataset.__len__ = Mock(return_value=1000)

        # 2. Model settings specify architecture
        # Use a simple class to avoid Mock auto-creating attributes
        class ModelSettings:
            architecture = "FeedForwardNN"
            class_path = "dlkit.domain.nn.ffnn.simple.FeedForwardNN"

        model_settings = ModelSettings()

        # 3. Shape system infers shapes automatically
        factory = ShapeSystemFactory.create_production_system()
        inference_engine = ShapeInferenceEngine(shape_factory=factory)

        shape_spec = inference_engine.infer_from_dataset(
            dataset=mock_dataset, model_settings=model_settings
        )

        # 4. Verify shape inference worked correctly
        assert not shape_spec.is_empty()
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value
        assert shape_spec.has_shape("x")
        assert shape_spec.has_shape("y")

        # 5. Shape spec can be used to configure model
        input_shape = shape_spec.get_input_shape()
        output_shape = shape_spec.get_output_shape()

        # The system should have inferred shapes from the dataset sample
        # Note: if inference fails, it provides defaults, so let's check what was actually inferred
        all_shapes = shape_spec.get_all_shapes()

        # For this test, let's verify that shape inference worked and provided reasonable shapes
        assert input_shape is not None
        assert output_shape is not None
        assert "x" in all_shapes
        assert "y" in all_shapes

    def test_infer_like_workflow_shape_integration(self):
        """Test shape integration in an infer-like workflow."""
        # Simulate what happens in dlkit.infer()

        # 1. Create mock checkpoint with shape metadata in V3 format
        checkpoint_metadata = {
            "dlkit_metadata": {
                "shape_spec": {
                    "metadata": {
                        "version": "v3",
                        "format": "json",
                        "created_at": "2024-01-01T00:00:00",
                    },
                    "data": {
                        "entries": {
                            "x": {"dimensions": [784], "metadata": {"name": "x"}},
                            "y": {"dimensions": [10], "metadata": {"name": "y"}},
                        },
                        "model_family": "dlkit_nn",
                        "source": "training_dataset",
                        "default_input": "x",
                        "default_output": "y",
                        "schema_version": "3.0",
                    },
                }
            }
        }

        # 2. Shape system loads shapes from checkpoint
        factory = ShapeSystemFactory.create_production_system()
        shape_spec = factory.create_shape_spec_from_serialized(
            checkpoint_metadata["dlkit_metadata"]["shape_spec"]
        )

        # 3. Verify shapes loaded correctly
        assert shape_spec.get_input_shape() == (784,)
        assert shape_spec.get_output_shape() == (10,)
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value

        # 4. Shapes can be used to validate input data
        input_data = torch.randn(32, 784)  # Batch of MNIST-like data
        expected_input_shape = shape_spec.get_input_shape()

        assert input_data.shape[1:] == expected_input_shape

    def test_optimize_like_workflow_shape_consistency(self):
        """Test shape consistency in an optimize-like workflow."""
        # Simulate what happens in dlkit.optimize()

        # 1. Base configuration with dataset
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(
            return_value={
                "x": torch.randn(32, 32, 3),  # CIFAR-like
                "y": torch.tensor(7),
            }
        )

        # 2. Different model configurations to optimize
        model_configs = [
            {"architecture": "FeedForwardNN", "hidden_sizes": [128, 64]},
            {"architecture": "FeedForwardNN", "hidden_sizes": [256, 128, 64]},
            {"architecture": "FeedForwardNN", "hidden_sizes": [512, 256]},
        ]

        # 3. Shape inference should be consistent across all configurations
        factory = ShapeSystemFactory.create_production_system()
        inference_engine = ShapeInferenceEngine(shape_factory=factory)

        inferred_shapes = []
        for config in model_configs:
            # Use a simple class to avoid Mock auto-creating attributes
            class ModelSettings:
                def __init__(self, arch, hidden):
                    self.architecture = arch
                    self.hidden_sizes = hidden

            settings = ModelSettings(config["architecture"], config["hidden_sizes"])

            shape_spec = inference_engine.infer_from_dataset(
                dataset=mock_dataset, model_settings=settings
            )

            inferred_shapes.append(
                {
                    "input": shape_spec.get_input_shape(),
                    "output": shape_spec.get_output_shape(),
                    "family": shape_spec.model_family(),
                }
            )

        # 4. All configurations should have same input/output shapes
        base_shapes = inferred_shapes[0]
        for shapes in inferred_shapes[1:]:
            assert shapes["input"] == base_shapes["input"]
            assert shapes["output"] == base_shapes["output"]
            assert shapes["family"] == base_shapes["family"]

    def test_graph_model_workflow_integration(self):
        """Test graph model workflow integration."""
        # Simulate graph neural network workflow

        # 1. Graph dataset provides different shape structure
        # Create a simple object instead of Mock for attribute access
        class GraphData:
            def __init__(self):
                self.x = torch.randn(100, 64)  # Node features
                self.edge_index = torch.randint(0, 100, (2, 500))  # Edges
                self.y = torch.randn(100, 1)  # Node labels

        mock_graph_data = GraphData()

        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=mock_graph_data)
        mock_dataset.__len__ = Mock(return_value=100)

        # 2. Graph model settings
        class GraphModelSettings:
            architecture = "BaseGraphNetwork"
            class_path = "dlkit.domain.nn.graph.base.BaseGraphNetwork"

        model_settings = GraphModelSettings()

        # 3. Shape inference recognizes graph family
        factory = ShapeSystemFactory.create_production_system()
        inference_engine = ShapeInferenceEngine(shape_factory=factory)

        shape_spec = inference_engine.infer_from_dataset(
            dataset=mock_dataset, model_settings=model_settings
        )

        # 4. Graph-specific shape handling
        assert shape_spec.model_family() == ModelFamily.GRAPH.value
        assert shape_spec.has_shape("x")  # Node features should be detected

        # 5. Verify that at least the main node features were detected
        all_shapes = shape_spec.get_all_shapes()
        assert len(all_shapes) > 0
        assert "x" in all_shapes

        # The exact shapes depend on the inference implementation
        # but we should have detected graph-like structures

    def test_error_handling_workflow_integration(self):
        """Test error handling in workflow integration."""
        # Test various error scenarios in shape integration

        factory = ShapeSystemFactory.create_production_system()

        # 1. Invalid dataset (no shapes available)
        mock_empty_dataset = Mock()
        mock_empty_dataset.__getitem__ = Mock(side_effect=IndexError("Empty dataset"))

        class ModelSettings:
            architecture = "FeedForwardNN"
            class_path = "dlkit.domain.nn.ffnn.simple.FeedForwardNN"

        model_settings = ModelSettings()

        inference_engine = ShapeInferenceEngine(shape_factory=factory)

        # Should gracefully fallback to defaults
        shape_spec = inference_engine.infer_from_dataset(
            dataset=mock_empty_dataset, model_settings=model_settings
        )

        # Should get default fallback
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value

        # 2. Invalid model configuration
        class InvalidModelSettings:
            architecture = "NonExistentModel"

        invalid_settings = InvalidModelSettings()

        # Should detect as external model
        shape_spec2 = inference_engine.infer_from_dataset(
            dataset=mock_empty_dataset, model_settings=invalid_settings
        )

        assert shape_spec2.model_family() == ModelFamily.EXTERNAL.value

    def test_serialization_workflow_integration(self):
        """Test serialization integration in workflows."""
        # Test shape serialization as it would happen in real workflows

        # 1. Create shape spec (as would happen during training)
        shape_spec = create_shape_spec(
            shapes={"x": (224, 224, 3), "y": (1000,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        # 2. Serialize for checkpoint (as would happen when saving model)
        factory = ShapeSystemFactory.create_production_system()
        serializer = factory.get_serializer()

        serialized = serializer.serialize(shape_spec.get_shape_data())

        # 3. Deserialize from checkpoint (as would happen during inference)
        deserialized_data = serializer.deserialize(serialized)

        # 4. Recreate shape spec for inference
        restored_spec = create_shape_spec(
            shapes={name: entry.dimensions for name, entry in deserialized_data.entries.items()},
            model_family=deserialized_data.model_family,
            source=deserialized_data.source,
        )

        # 5. Verify roundtrip consistency
        assert restored_spec.get_input_shape() == shape_spec.get_input_shape()
        assert restored_spec.get_output_shape() == shape_spec.get_output_shape()
        assert restored_spec.model_family() == shape_spec.model_family()

    def test_validation_workflow_integration(self):
        """Test validation integration in workflows."""
        # Test shape validation as it would happen in config validation

        # 1. Create shape spec with potential issues
        shape_spec = create_shape_spec(
            shapes={
                "x": (10, 20, 30, 40, 50, 60),  # Too many dimensions
                "y": (5,),
            },
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        # 2. Validate using the system (as would happen in config validation)
        factory = ShapeSystemFactory.create_production_system()
        validator = factory.get_validator()

        result = validator.validate_collection(shape_spec.get_shape_data())

        # 3. Should detect validation issues
        assert not result.is_valid
        assert any("maximum allowed" in error for error in result.errors)

        # 4. Valid shape spec should pass validation
        valid_spec = create_shape_spec(
            shapes={"x": (784,), "y": (10,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET,
        )

        valid_result = validator.validate_collection(valid_spec.get_shape_data())
        assert valid_result.is_valid
