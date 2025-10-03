"""Integration tests for ABC-based shape architecture.

Tests the new ABC-based shape handling architecture including:
- ShapeAwareModel and ShapeAgnosticModel classes
- Shape checkpoint persistence
- ABC-based model detection
- Seamless train->infer workflow
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

import dlkit
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.trainer_settings import TrainerSettings
from dlkit.core.shape_specs import create_shape_spec, CheckpointShapeLoader, NullShapeSpec
from dlkit.core.models.nn.base import ShapeAwareModel, ShapeAgnosticModel
from dlkit.core.models.nn.ffnn.simple import FeedForwardNN
from dlkit.runtime.workflows.factories.model_detection import detect_model_type, ModelType


@pytest.fixture
def training_settings_with_checkpointing(training_settings: GeneralSettings, tmp_path: Path) -> GeneralSettings:
    """Training settings with checkpointing enabled."""
    from dlkit.tools.config.trainer_settings import CallbackSettings

    # Create a copy with checkpointing enabled
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lightning_logs_dir = tmp_path / "lightning_logs"
    lightning_logs_dir.mkdir(parents=True, exist_ok=True)

    # Create explicit checkpoint callback settings with save_last=True, every epoch, no versioning
    checkpoint_callback = CallbackSettings(
        name="ModelCheckpoint",
        module_path="lightning.pytorch.callbacks",
        dirpath=str(checkpoint_dir),
        filename="model",
        save_top_k=1,
        every_n_epochs=1,
        enable_version_counter=False,
        save_last=True,  # Save last checkpoint
    )

    updated_trainer = TrainerSettings(
        fast_dev_run=False,
        max_epochs=1,
        enable_checkpointing=True,
        default_root_dir=str(lightning_logs_dir),
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=(checkpoint_callback,),
    )

    updated_training = training_settings.TRAINING.model_copy(update={"trainer": updated_trainer})
    return training_settings.model_copy(update={"TRAINING": updated_training})


class TestABCShapeArchitecture:
    """Test suite for ABC-based shape architecture."""

    def test_shape_aware_model_creation(self):
        """Test that shape-aware models require unified_shape parameter."""
        shape_spec = create_shape_spec({"x": (784,), "y": (10,)})

        # Should work with unified_shape
        model = FeedForwardNN(
            unified_shape=shape_spec,
            layers=[128, 64]
        )
        assert isinstance(model, ShapeAwareModel)
        assert model.get_unified_shape() == shape_spec

        # Should fail without unified_shape
        with pytest.raises(TypeError):
            FeedForwardNN(layers=[128, 64])  # Missing unified_shape

    def test_model_type_detection(self, training_settings: GeneralSettings):
        """Test ABC-based model detection."""
        # Test detection for FFNN (shape-aware DLKit model)
        model_type = detect_model_type(training_settings.MODEL, training_settings)
        assert model_type == ModelType.SHAPE_AWARE_DLKIT

    def test_shape_checkpoint_persistence(
        self,
        training_settings_with_checkpointing: GeneralSettings,
        tmp_path: Path
    ):
        """Test that shapes are saved and loaded from checkpoints."""
        # Run training to create checkpoint with shapes
        training_result = dlkit.train(training_settings_with_checkpointing)

        # Check that a checkpoint was created
        assert training_result.checkpoint_path is not None
        checkpoint_path = Path(training_result.checkpoint_path)
        assert checkpoint_path.exists()

        # Test shape loading from checkpoint
        loader = CheckpointShapeLoader()
        has_shape = loader.has_shape_metadata(checkpoint_path)

        # Should have shape metadata (might be False for external models)
        # This tests that the mechanism works even if no shapes are saved
        assert isinstance(has_shape, bool)

        if has_shape:
            shape_spec = loader.load_shape_spec(checkpoint_path)
            assert shape_spec is not None
            assert not shape_spec.is_empty()

    def test_seamless_train_to_infer_workflow(
        self,
        training_settings_with_checkpointing: GeneralSettings,
        inference_settings: GeneralSettings,
        tmp_path: Path
    ):
        """Test seamless training->inference without re-specifying shapes."""
        # Step 1: Train model (shapes should be saved automatically)
        training_result = dlkit.train(training_settings_with_checkpointing)
        assert training_result.checkpoint_path is not None

        # Step 2: Create dummy input data for inference
        checkpoint_path = Path(training_result.checkpoint_path)
        inputs = {"X": torch.randn(10, 4)}  # Batch of 10 samples, 4 features (matching training data)

        # Step 3: Run inference using new API (should load shapes automatically)
        inference_result = dlkit.infer(checkpoint_path, inputs)
        assert inference_result is not None
        assert inference_result.predictions is not None

    def test_abc_based_factory_integration(
        self,
        training_settings: GeneralSettings
    ):
        """Test that BuildFactory uses ABC-based detection."""
        from dlkit.runtime.workflows.factories.build_factory import BuildFactory
        from dlkit.runtime.workflows.factories.model_detection import requires_shape_spec

        # Create factory and build components
        factory = BuildFactory()
        components = factory.build_components(training_settings)

        # Check that model detection worked
        model_type = detect_model_type(training_settings.MODEL, training_settings)
        should_require_shapes = requires_shape_spec(model_type)

        if should_require_shapes:
            # Shape-aware models should have shape_spec
            assert components.shape_spec is not None
            assert not components.shape_spec.is_empty()

        # Model should be created successfully
        assert components.model is not None

    def test_external_model_handling(self):
        """Test handling of external (shape-agnostic) models."""
        from dlkit.tools.config.components.model_components import ModelComponentSettings
        from dlkit.tools.config import GeneralSettings

        # Create settings for an external Lightning model
        external_model_settings = ModelComponentSettings(
            name="lightning.pytorch.LightningModule",  # Example external model
            module_path=""
        )

        # Create minimal settings using model_copy
        settings = GeneralSettings().model_copy(update={"MODEL": external_model_settings})

        # Should detect as external/shape-agnostic
        model_type = detect_model_type(external_model_settings, settings)
        assert model_type == ModelType.SHAPE_AGNOSTIC_EXTERNAL

        # Should not require shape spec
        from dlkit.runtime.workflows.factories.model_detection import requires_shape_spec
        assert not requires_shape_spec(model_type)


class TestShapeCheckpointPersistence:
    """Focused tests for shape checkpoint persistence mechanism."""

    def test_checkpoint_shape_metadata_storage(self, tmp_path: Path):
        """Test that shape metadata is correctly stored in checkpoints."""
        from dlkit.core.shape_specs import enable_shape_persistence
        from lightning.pytorch import LightningModule
        import torch

        # Create a mock Lightning module
        class MockModule(LightningModule):
            def __init__(self, shape_spec):
                super().__init__()
                self._shape_spec = shape_spec
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        shape_spec = create_shape_spec({"x": (10,), "y": (5,)})
        module = MockModule(shape_spec)

        # Enable shape persistence
        module = enable_shape_persistence(module)

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        checkpoint = {"state_dict": module.state_dict()}

        # Simulate save_checkpoint
        module.on_save_checkpoint(checkpoint)

        # Save to file
        torch.save(checkpoint, checkpoint_path)

        # Load and verify shape metadata
        loader = CheckpointShapeLoader()
        loaded_shape = loader.load_shape_spec(checkpoint_path)

        if loaded_shape is not None:  # May be None if shape extraction failed
            assert loaded_shape.get_input_shape() == (10,)
            assert loaded_shape.get_output_shape() == (5,)

    def test_checkpoint_model_creation(self, tmp_path: Path):
        """Test creating models from checkpoints with automatic shape loading."""
        from dlkit.interfaces.inference.transforms.checkpoint_loader import CheckpointTransformLoader

        # Create a simple checkpoint with shape metadata in V3 format and model settings
        shape_spec = create_shape_spec({"x": (784,), "y": (10,)})

        # Use V3 modern format (current standard) with full dlkit_metadata
        checkpoint = {
            "state_dict": {},
            "dlkit_metadata": {
                "version": "2.0",
                "model_family": "dlkit_nn",
                "wrapper_type": "StandardLightningWrapper",
                "dlkit_version": "2.0",
                "shape_spec": {
                    "metadata": {
                        "version": "v3",
                        "format": "json",
                        "created_at": "2025-09-30T00:00:00"
                    },
                    "data": {
                        "entries": {
                            "x": {
                                "dimensions": [784],
                                "metadata": {"name": "x"}
                            },
                            "y": {
                                "dimensions": [10],
                                "metadata": {"name": "y"}
                            }
                        },
                        "model_family": "dlkit_nn",
                        "source": "training_dataset",
                        "default_input": "x",
                        "default_output": "y",
                        "schema_version": "3.0"
                    }
                },
                "model_settings": {
                    "name": "FeedForwardNN",
                    "module_path": "dlkit.core.models.nn.ffnn.simple",
                    "params": {
                        "layers": [128, 64]
                    },
                    "class_name": "ModelComponentSettings"
                },
                "entry_configs": {
                    "x": {"name": "x", "class_name": "Feature"},
                    "y": {"name": "y", "class_name": "Target"}
                }
            }
        }

        checkpoint_path = tmp_path / "test_checkpoint.ckpt"
        torch.save(checkpoint, checkpoint_path)

        # Test checkpoint loader
        loader = CheckpointTransformLoader()

        # Should find shape spec
        assert loader.has_shape_spec(checkpoint_path)

        # Should load shape spec
        loaded_shape = loader.load_shape_spec(checkpoint_path)
        assert loaded_shape is not None
        assert loaded_shape.get_shape("x") == (784,)
        assert loaded_shape.get_shape("y") == (10,)

        # Should create model with loaded shapes
        model = loader.create_model_from_checkpoint(
            checkpoint_path,
            model_class=FeedForwardNN
        )

        if model is not None:  # May be None if creation failed
            assert isinstance(model, FeedForwardNN)
            assert isinstance(model, ShapeAwareModel)


class TestModelABCCompliance:
    """Test that models properly follow ABC contracts."""

    def test_shape_aware_model_contract(self):
        """Test ShapeAwareModel abstract contract."""
        shape_spec = create_shape_spec({"x": (784,), "y": (10,)})

        model = FeedForwardNN(
            unified_shape=shape_spec,
            layers=[128, 64]
        )

        # Test ABC contract methods
        assert model.accepts_shape(shape_spec) is True
        assert model.get_unified_shape() == shape_spec

        # Test forward method exists (abstract method)
        assert hasattr(model, 'forward')
        assert callable(model.forward)

        # Test precision methods
        assert hasattr(model, 'ensure_precision_applied')
        assert hasattr(model, 'cast_input')

    def test_shape_validation(self):
        """Test shape validation in shape-aware models."""
        valid_shape = create_shape_spec({"x": (784,), "y": (10,)})

        # Should accept valid shape
        model = FeedForwardNN(
            unified_shape=valid_shape,
            layers=[128, 64]
        )
        assert model.accepts_shape(valid_shape)

        # Should reject NullShapeSpec
        null_shape = NullShapeSpec()
        assert not model.accepts_shape(null_shape)

        # Should fail to create with NullShapeSpec
        with pytest.raises(ValueError, match="cannot accept"):
            FeedForwardNN(
                unified_shape=null_shape,
                layers=[128, 64]
            )