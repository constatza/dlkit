"""Integration tests for simplified predictor architecture.

Tests the core predictor functionality after Phase 1 refactoring:
- Consolidated 27 files → 6 files
- Direct loading without hexagonal architecture
- Stateful predictor pattern
- Precision inference
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from dlkit.engine.inference.loading import (
    build_model_from_checkpoint,
    detect_checkpoint_dtype,
    extract_state_dict,
    get_checkpoint_info,
    load_checkpoint,
    validate_checkpoint,
)
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from dlkit.interfaces.inference import CheckpointPredictor, PredictorConfig, load_model


class TestCheckpointLoading:
    """Test checkpoint loading utilities."""

    def test_load_checkpoint(self, tmp_path: Path):
        """Test basic checkpoint loading."""
        # Create simple checkpoint
        checkpoint_path = tmp_path / "model.ckpt"
        model = torch.nn.Linear(10, 5)
        checkpoint = {"state_dict": model.state_dict(), "dlkit_metadata": {}}
        torch.save(checkpoint, checkpoint_path)

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path)

        assert isinstance(loaded, dict)
        assert "state_dict" in loaded
        assert "dlkit_metadata" in loaded

    def test_extract_state_dict_strips_prefix(self):
        """Test that extract_state_dict strips model. prefix."""
        checkpoint = {
            "state_dict": {
                "model.weight": torch.randn(5, 10),
                "model.bias": torch.randn(5),
            }
        }

        state_dict = extract_state_dict(checkpoint)

        assert "weight" in state_dict
        assert "bias" in state_dict
        assert "model.weight" not in state_dict

    def test_detect_checkpoint_dtype_float32(self):
        """Test dtype detection for float32 checkpoint."""
        state_dict = {
            "weight": torch.randn(10, 5, dtype=torch.float32),
            "bias": torch.randn(5, dtype=torch.float32),
        }

        dtype = detect_checkpoint_dtype(state_dict)

        assert dtype == torch.float32

    def test_detect_checkpoint_dtype_float64(self):
        """Test dtype detection for float64 checkpoint."""
        state_dict = {
            "weight": torch.randn(10, 5, dtype=torch.float64),
            "bias": torch.randn(5, dtype=torch.float64),
        }

        dtype = detect_checkpoint_dtype(state_dict)

        assert dtype == torch.float64

    def test_validate_checkpoint_valid(self, tmp_path: Path):
        """Test checkpoint validation with valid checkpoint."""
        checkpoint_path = tmp_path / "model.ckpt"
        model = torch.nn.Linear(10, 5)
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {
                "model_settings": {
                    "name": "Linear",
                    "module_path": "torch.nn",
                    "in_features": 10,
                    "out_features": 5,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)

        results = validate_checkpoint(checkpoint_path)

        # Test dataclass attributes (not dict subscripting)
        assert results.exists is True
        assert results.valid_format is True
        assert results.has_state_dict is True
        assert results.has_model_settings is True

    def test_get_checkpoint_info(self, tmp_path: Path):
        """Test checkpoint info extraction."""
        checkpoint_path = tmp_path / "model.ckpt"
        model = torch.nn.Linear(10, 5)
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {"model_family": "ffnn"},
        }
        torch.save(checkpoint, checkpoint_path)

        info = get_checkpoint_info(checkpoint_path)

        # Test dataclass attributes (not dict subscripting)
        assert info.has_dlkit_metadata is True
        assert info.model_family == "ffnn"


class TestBuildModelFromCheckpoint:
    """Test model building from checkpoint."""

    def test_build_simple_model(self, tmp_path: Path):
        """Test building a simple PyTorch model from checkpoint."""
        # Create checkpoint with model settings
        model = torch.nn.Linear(10, 5).to(torch.float32)
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {
                "model_settings": {
                    "name": "Linear",
                    "module_path": "torch.nn",
                    "in_features": 10,
                    "out_features": 5,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Load and build model — None shape means external model, uses kwargs only
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        built_model = build_model_from_checkpoint(loaded_checkpoint, None)

        # Verify model
        assert isinstance(built_model, torch.nn.Linear)
        assert built_model.in_features == 10
        assert built_model.out_features == 5
        assert not built_model.training  # Should be in eval mode

    def test_build_model_preserves_dtype(self, tmp_path: Path):
        """Test that model dtype matches checkpoint dtype."""
        # Create float64 model
        model = torch.nn.Linear(10, 5).to(torch.float64)
        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {
                "model_settings": {
                    "name": "Linear",
                    "module_path": "torch.nn",
                    "in_features": 10,
                    "out_features": 5,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)

        # Build model — None shape means external model, uses kwargs only
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        built_model = build_model_from_checkpoint(loaded_checkpoint, None)

        # Verify dtype matches checkpoint
        assert built_model.weight.dtype == torch.float64


class TestCheckpointPredictor:
    """Test CheckpointPredictor class."""

    @pytest.fixture
    def simple_checkpoint(self, tmp_path: Path) -> Path:
        """Create a simple checkpoint for testing."""
        model = torch.nn.Linear(10, 5).to(torch.float32)
        model.eval()

        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {
                "model_settings": {
                    "name": "Linear",
                    "module_path": "torch.nn",
                    "in_features": 10,
                    "out_features": 5,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_predictor_loads_model(self, simple_checkpoint: Path):
        """Test that predictor loads model successfully."""
        config = PredictorConfig(checkpoint_path=simple_checkpoint, auto_load=False)

        predictor = CheckpointPredictor(config)
        assert not predictor.is_loaded()

        predictor.load()

        assert predictor.is_loaded()
        assert predictor._model_state is not None
        assert predictor._model_state.model is not None

    def test_predictor_auto_load(self, simple_checkpoint: Path):
        """Test predictor with auto_load=True."""
        config = PredictorConfig(checkpoint_path=simple_checkpoint, auto_load=True)

        predictor = CheckpointPredictor(config)

        # Should be loaded automatically
        assert predictor.is_loaded()

    def test_predictor_infers_precision(self, simple_checkpoint: Path):
        """Test that predictor infers precision from model."""
        config = PredictorConfig(checkpoint_path=simple_checkpoint, auto_load=True)

        predictor = CheckpointPredictor(config)

        # Should infer float32 from model
        assert predictor._inferred_precision == PrecisionStrategy.FULL_32

    def test_predictor_predict_positional(self, simple_checkpoint: Path):
        """Test prediction with a positional tensor arg."""
        config = PredictorConfig(
            checkpoint_path=simple_checkpoint,
            auto_load=True,
            apply_transforms=False,  # No transforms in simple checkpoint
        )

        predictor = CheckpointPredictor(config)

        # Positional tensor input — mirrors model.forward(tensor)
        result = predictor.predict(torch.randn(32, 10))

        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 5)

    def test_predictor_predict_kwarg(self, simple_checkpoint: Path):
        """Test prediction with a keyword tensor arg."""
        config = PredictorConfig(
            checkpoint_path=simple_checkpoint, auto_load=True, apply_transforms=False
        )

        predictor = CheckpointPredictor(config)

        # Kwarg input — mirrors model.forward(weight=tensor)
        result = predictor.predict(input=torch.randn(32, 10))

        assert isinstance(result, torch.Tensor)
        assert result.shape == (32, 5)

    def test_predictor_context_manager(self, simple_checkpoint: Path):
        """Test predictor as context manager."""
        config = PredictorConfig(checkpoint_path=simple_checkpoint, auto_load=False)

        with CheckpointPredictor(config) as predictor:
            assert predictor.is_loaded()
            predictions = predictor.predict(torch.randn(32, 10))
            assert isinstance(predictions, torch.Tensor)
            assert predictions.shape == (32, 5)

        # Should be unloaded after context exit
        assert not predictor.is_loaded()

    def test_predictor_unload(self, simple_checkpoint: Path):
        """Test predictor unload."""
        config = PredictorConfig(checkpoint_path=simple_checkpoint, auto_load=True)

        predictor = CheckpointPredictor(config)
        assert predictor.is_loaded()

        predictor.unload()

        assert not predictor.is_loaded()
        assert predictor._model_state is None


class TestLoadPredictorAPI:
    """Test the public load_model() API."""

    @pytest.fixture
    def simple_checkpoint(self, tmp_path: Path) -> Path:
        """Create a simple checkpoint for testing."""
        model = torch.nn.Linear(10, 5).to(torch.float32)
        model.eval()

        checkpoint_path = tmp_path / "model.ckpt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "dlkit_metadata": {
                "model_settings": {
                    "name": "Linear",
                    "module_path": "torch.nn",
                    "in_features": 10,
                    "out_features": 5,
                },
            },
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_load_model_simple(self, simple_checkpoint: Path):
        """Test load_model() factory function."""
        predictor = load_model(simple_checkpoint, device="cpu")

        assert isinstance(predictor, CheckpointPredictor)
        assert predictor.is_loaded()

        # predict() returns Tensor directly for single-output models
        result = predictor.predict(torch.randn(16, 10))

        assert isinstance(result, torch.Tensor)
        assert result.shape == (16, 5)

    def test_load_model_with_precision(self, simple_checkpoint: Path):
        """Test load_model() with precision parameter."""
        predictor = load_model(simple_checkpoint, device="cpu", precision=PrecisionStrategy.FULL_64)

        # Config precision should override inferred precision
        assert predictor._config.precision == PrecisionStrategy.FULL_64

    def test_load_model_context_manager(self, simple_checkpoint: Path):
        """Test load_model() with context manager."""
        with load_model(simple_checkpoint) as predictor:
            result = predictor.predict(torch.randn(8, 10))
            assert isinstance(result, torch.Tensor)
            assert result.shape == (8, 5)

    def test_predictor_preserves_tuple_model_outputs(self, simple_checkpoint: Path):
        """Tuple-valued forward outputs remain supported after Batch cleanup."""

        class TupleOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return x[:, :5], x[:, 5:]

        predictor = load_model(simple_checkpoint, device="cpu")
        assert predictor._model_state is not None

        predictor._model_state = replace(
            predictor._model_state,
            model=TupleOutputModel().eval(),
        )

        result = predictor.predict(torch.randn(8, 10))

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (8, 5)
        assert result[1].shape == (8, 5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
