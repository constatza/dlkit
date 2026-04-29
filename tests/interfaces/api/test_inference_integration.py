"""Integration tests for the predictor-based inference API.

NOTE: Core predictor functionality is comprehensively tested in
tests/interfaces/inference/test_simplified_predictor.py (18 tests).

This file focuses on API imports.
"""

from __future__ import annotations


class TestPredictorAPIImports:
    """Test that predictor API can be imported correctly."""

    def test_import_load_model(self):
        """Test importing load_model from main package."""
        from dlkit import load_model

        assert callable(load_model)

    def test_import_predictor_classes(self):
        """Test importing predictor classes."""
        from dlkit.interfaces.inference import (
            CheckpointPredictor,
            IPredictor,
            PredictionOutput,
            PredictorConfig,
        )

        assert CheckpointPredictor is not None
        assert IPredictor is not None
        assert PredictionOutput is not None
        assert PredictorConfig is not None

    def test_import_utilities(self):
        """Test importing utility functions."""
        from dlkit.interfaces.inference import (
            get_checkpoint_info,
            load_model_from_settings,
            validate_checkpoint,
        )

        assert callable(validate_checkpoint)
        assert callable(get_checkpoint_info)
        assert callable(load_model_from_settings)
