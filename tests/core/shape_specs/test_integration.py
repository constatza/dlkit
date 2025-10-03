"""Integration tests for the complete shape handling system.

This module tests the integration of all components working together
in realistic scenarios.
"""

import pytest
from pathlib import Path
import tempfile
import json
import torch
from unittest.mock import Mock, patch

from dlkit.core.shape_specs import (
    create_shape_spec, ShapeSystemFactory, ModelFamily, ShapeSource,
    ShapeInferenceEngine, InferenceContext, CachingShapeInferencer,
    ShapeInferenceChain, BatchShapeProcessor, VersionedShapeSerializer,
    SerializationFormat, ShapeData, ShapeEntry
)
from dlkit.tools.config.components.model_components import ModelComponentSettings


class TestFullSystemIntegration:
    """Test complete system integration scenarios."""

    @pytest.fixture
    def shape_factory(self):
        """Shape system factory for testing."""
        return ShapeSystemFactory.create_testing_system()

    @pytest.fixture
    def sample_dataset(self):
        """Mock dataset for testing."""
        # Create mock dataset with dict-based samples
        dataset = Mock()
        dataset.__getitem__ = Mock(return_value={
            'x': torch.randn(10, 20),
            'y': torch.randn(5),
            'features': torch.randn(100, 50)
        })
        dataset.__len__ = Mock(return_value=1000)
        return dataset

    @pytest.fixture
    def sample_model_settings(self):
        """Mock model settings for testing."""
        settings = Mock()
        settings.architecture = "FeedForwardNN"
        settings.class_path = "dlkit.core.models.nn.ffnn.simple.FeedForwardNN"
        return settings

    def test_end_to_end_shape_inference_from_dataset(self, shape_factory, sample_dataset, sample_model_settings):
        """Test complete end-to-end shape inference from dataset."""
        # Create inference engine
        inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)

        # Infer shapes from dataset
        shape_spec = inference_engine.infer_from_dataset(
            dataset=sample_dataset,
            model_settings=sample_model_settings
        )

        # Verify results
        assert not shape_spec.is_empty()
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value
        assert shape_spec.has_shape("x")
        assert shape_spec.has_shape("y")
        assert shape_spec.get_shape("x") == (10, 20)
        assert shape_spec.get_shape("y") == (5,)

    def test_shape_inference_with_real_model_settings(self, shape_factory):
        """Ensure end-to-end inference works with concrete model settings."""

        class TinyDataset:
            def __len__(self):  # pragma: no cover - trivial
                return 8

            def __getitem__(self, index):  # pragma: no cover - trivial
                return {
                    "x": torch.zeros(32),
                    "y": torch.zeros(4),
                }

        dataset = TinyDataset()
        model_settings = ModelComponentSettings(
            name="LinearNetwork",
            module_path="dlkit.core.models.nn.ffnn.linear",
            class_path="dlkit.core.models.nn.ffnn.linear.LinearNetwork",
        )

        inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)
        shape_spec = inference_engine.infer_from_dataset(
            dataset=dataset,
            model_settings=model_settings,
        )

        assert not shape_spec.is_empty()
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value
        assert shape_spec.get_input_shape() == (32,)
        assert shape_spec.get_output_shape() == (4,)

    def test_end_to_end_checkpoint_serialization_roundtrip(self, shape_factory):
        """Test complete checkpoint serialization and deserialization."""
        # Create original shape spec
        original_spec = create_shape_spec(
            shapes={"x": (784,), "y": (10,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET
        )

        # Serialize for checkpoint
        serializer = VersionedShapeSerializer()
        serialized = serializer.serialize(original_spec.get_shape_data())

        # Create mock checkpoint
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pth', delete=False) as f:
            checkpoint_path = Path(f.name)

        try:
            # Save checkpoint with shape metadata
            checkpoint_data = {
                'model_state_dict': {},
                'dlkit_metadata': {
                    'shape_spec': serialized.to_dict()
                }
            }
            torch.save(checkpoint_data, checkpoint_path)

            # Create inference engine and infer from checkpoint
            inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)
            reconstructed_spec = inference_engine.infer_from_checkpoint(checkpoint_path)

            # Verify roundtrip
            assert reconstructed_spec.get_shape("x") == (784,)
            assert reconstructed_spec.get_shape("y") == (10,)
            assert reconstructed_spec.model_family() == ModelFamily.DLKIT_NN.value

        finally:
            checkpoint_path.unlink()

    def test_caching_performance_integration(self, shape_factory, sample_dataset, sample_model_settings):
        """Test caching integration with performance benefits."""
        # Create base inference chain
        base_chain = ShapeInferenceChain()

        # Create caching wrapper
        caching_inferencer = CachingShapeInferencer(base_chain)

        # Create inference context
        context = InferenceContext(
            dataset=sample_dataset,
            model_settings=sample_model_settings,
            shape_factory=shape_factory
        )

        # First inference (cache miss)
        first_result = caching_inferencer.infer_shape_spec(context)
        stats_after_first = caching_inferencer.get_cache_stats()

        assert stats_after_first.total_requests == 1
        assert stats_after_first.misses == 1
        assert stats_after_first.hits == 0

        # Second inference with same context (cache hit)
        second_result = caching_inferencer.infer_shape_spec(context)
        stats_after_second = caching_inferencer.get_cache_stats()

        assert stats_after_second.total_requests == 2
        assert stats_after_second.hits == 1
        assert stats_after_second.hit_rate > 0

        # Results should be identical
        assert first_result.get_all_shapes() == second_result.get_all_shapes()

    def test_batch_processing_integration(self, shape_factory):
        """Test batch processing integration."""
        # Create multiple shape data objects
        shape_data_list = []
        for i in range(5):
            entries = {
                "x": ShapeEntry(name="x", dimensions=(10 * (i + 1), 20)),
                "y": ShapeEntry(name="y", dimensions=(5,))
            }
            shape_data = ShapeData(
                entries=entries,
                model_family=ModelFamily.DLKIT_NN,
                source=ShapeSource.TRAINING_DATASET
            )
            shape_data_list.append(shape_data)

        # Create batch processor
        batch_processor = shape_factory.get_batch_processor()

        # Batch validate
        validation_results = batch_processor.validate_batch(shape_data_list)
        assert len(validation_results) == 5
        assert all(result.is_valid for result in validation_results)

        # Batch serialize
        serialized_list = batch_processor.serialize_batch(shape_data_list)
        assert len(serialized_list) == 5

        # Batch deserialize
        deserialized_list = batch_processor.deserialize_batch(serialized_list)
        assert len(deserialized_list) == 5

        # Verify roundtrip
        for original, deserialized in zip(shape_data_list, deserialized_list):
            assert original.entries.keys() == deserialized.entries.keys()
            for key in original.entries:
                assert original.entries[key].dimensions == deserialized.entries[key].dimensions

    def test_registry_based_model_detection_integration(self, shape_factory):
        """Test registry-based model detection integration."""
        # Use production factory to get all detectors
        production_factory = ShapeSystemFactory.create_production_system()
        registry = production_factory.get_model_registry()

        # Test DLKIT NN detection
        class DLKitSettings:
            architecture = "FeedForwardNN"
            class_path = "dlkit.core.models.nn.ffnn.simple.FeedForwardNN"
        detected_family = registry.detect_family(DLKitSettings())
        assert detected_family == ModelFamily.DLKIT_NN

        # Test graph detection
        class GraphSettings:
            architecture = "BaseGraphNetwork"
            class_path = "dlkit.core.models.nn.graph.base.BaseGraphNetwork"
        detected_family = registry.detect_family(GraphSettings())
        assert detected_family == ModelFamily.GRAPH

        # Test external detection (class without MODEL_FAMILY attribute)
        class ExternalSettings:
            class_path = "pytorch_forecasting.models.DeepAR"
        detected_family = registry.detect_family(ExternalSettings())
        assert detected_family == ModelFamily.EXTERNAL

    def test_specification_validation_integration(self, shape_factory):
        """Test specification-based validation integration."""
        # Create shape data with validation issues
        entries = {
            "x": ShapeEntry(name="x", dimensions=(10, 20, 30, 40, 50, 60))  # Too many dimensions
        }
        shape_data = ShapeData(
            entries=entries,
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET
        )

        # Validate using factory validator
        validator = shape_factory.get_validator()
        result = validator.validate_collection(shape_data)

        # Should fail due to too many dimensions for DLKIT_NN
        assert not result.is_valid
        assert any("maximum allowed" in error for error in result.errors)

    def test_legacy_format_migration_integration(self, shape_factory):
        """Test legacy format migration integration."""
        # Create legacy shape_info format
        legacy_shape_info = {
            '_type': 'dict',
            'data': {
                'x': [784],
                'y': [10]
            }
        }

        # Migrate using factory
        serializer = shape_factory.get_serializer()
        migrated_data = serializer.deserialize_legacy_format(legacy_shape_info)

        assert migrated_data is not None
        assert migrated_data.has_entry("x")
        assert migrated_data.has_entry("y")
        assert migrated_data.get_dimensions("x") == (784,)
        assert migrated_data.get_dimensions("y") == (10,)

    def test_comprehensive_inference_fallback_chain(self, shape_factory, sample_model_settings):
        """Test comprehensive inference with fallback chain."""
        inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)

        # Test with no data sources (should fallback to defaults)
        shape_spec = inference_engine.infer_comprehensive(
            model_settings=sample_model_settings
        )

        # Should not be empty (fallback should provide defaults)
        assert not shape_spec.is_empty()
        assert shape_spec.model_family() == ModelFamily.DLKIT_NN.value

    def test_error_handling_integration(self, shape_factory):
        """Test error handling across system integration."""
        # Test invalid serialized data
        serializer = shape_factory.get_serializer()

        try:
            serializer.deserialize({"invalid": "data"})
            assert False, "Should have raised an exception"
        except (ValueError, KeyError):
            pass  # Expected behavior

        # Test invalid validation
        validator = shape_factory.get_validator()
        invalid_data = ShapeData(
            entries={},  # Empty entries
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET
        )

        result = validator.validate_collection(invalid_data)
        # Empty data might be valid for some specs, but should trigger warnings
        # The exact behavior depends on the specific validation rules

    @patch('torch.load')
    def test_checkpoint_loading_error_handling(self, mock_torch_load, shape_factory):
        """Test error handling during checkpoint loading."""
        # Mock torch.load to raise an exception
        mock_torch_load.side_effect = RuntimeError("Corrupted checkpoint")

        inference_engine = ShapeInferenceEngine(shape_factory=shape_factory)

        # Should fallback gracefully when checkpoint loading fails
        with tempfile.NamedTemporaryFile(suffix='.pth') as f:
            checkpoint_path = Path(f.name)
            shape_spec = inference_engine.infer_from_checkpoint(checkpoint_path)

            # Should provide fallback result
            assert shape_spec is not None
            # Should be external family as fallback
            assert shape_spec.model_family() == ModelFamily.EXTERNAL.value


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    @pytest.fixture
    def production_factory(self):
        """Production-configured factory."""
        return ShapeSystemFactory.create_production_system()

    def test_mnist_classification_scenario(self, production_factory):
        """Test MNIST classification scenario."""
        # Create MNIST-like shape spec
        shape_spec = create_shape_spec(
            shapes={"x": (784,), "y": (10,)},
            model_family=ModelFamily.DLKIT_NN,
            source=ShapeSource.TRAINING_DATASET
        )

        # Validate
        validator = production_factory.get_validator()
        result = validator.validate_collection(shape_spec.get_shape_data())
        assert result.is_valid

        # Serialize
        serializer = production_factory.get_serializer()
        serialized = serializer.serialize(shape_spec.get_shape_data())
        deserialized = serializer.deserialize(serialized)

        # Verify roundtrip
        assert deserialized.get_dimensions("x") == (784,)
        assert deserialized.get_dimensions("y") == (10,)

    def test_graph_node_classification_scenario(self, production_factory):
        """Test graph node classification scenario."""
        # Create graph shape spec
        shape_spec = create_shape_spec(
            shapes={
                "x": (100, 64),      # Node features
                "edge_index": (2, 5000),  # Edge connectivity
                "y": (100,)          # Node labels
            },
            model_family=ModelFamily.GRAPH,
            source=ShapeSource.GRAPH_DATASET
        )

        # Validate with graph-specific rules
        validator = production_factory.get_validator()
        result = validator.validate_collection(shape_spec.get_shape_data())
        assert result.is_valid

        # Verify it recognized as graph family
        assert shape_spec.model_family() == ModelFamily.GRAPH.value

    def test_external_model_scenario(self, production_factory):
        """Test external model scenario (no shapes)."""
        # Create external model spec
        shape_spec = create_shape_spec(
            shapes=None,
            model_family=ModelFamily.EXTERNAL,
            source=ShapeSource.CONFIGURATION
        )

        # Should be empty
        assert shape_spec.is_empty()
        assert shape_spec.model_family() == ModelFamily.EXTERNAL.value

        # Validation should pass with warnings
        validator = production_factory.get_validator()
        result = validator.validate_collection(shape_spec.get_shape_data())
        assert result.is_valid

    def test_time_series_forecasting_scenario(self, production_factory):
        """Test time series forecasting scenario."""
        # Create time series shape spec
        # Note: TIMESERIES is still a valid enum value for manual specification
        # PyTorch Forecasting models are auto-detected as EXTERNAL, but users can
        # manually specify TIMESERIES for shape validation purposes
        shape_spec = create_shape_spec(
            shapes={
                "x": (100, 24, 10),  # (batch, sequence, features)
                "y": (100, 1)        # (batch, prediction)
            },
            model_family=ModelFamily.TIMESERIES,
            source=ShapeSource.TRAINING_DATASET
        )

        # Validate
        validator = production_factory.get_validator()
        result = validator.validate_collection(shape_spec.get_shape_data())
        assert result.is_valid

        assert shape_spec.model_family() == ModelFamily.TIMESERIES.value
        assert len(shape_spec.get_shape("x")) == 3  # Multi-dimensional for time series


# Removed: test_model_family_detection_uses_module_path
# Pure class-based detection now requires fully qualified class_path
# No fallback to module_path or architecture name matching
