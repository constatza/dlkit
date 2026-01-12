"""Integration tests for the predictor-based inference API.

NOTE: Core predictor functionality is comprehensively tested in
tests/interfaces/inference/test_simplified_predictor.py (18 tests).

This file focuses on API imports.
"""

from __future__ import annotations


class TestPredictorAPIImports:
    """Test that predictor API can be imported correctly."""

    def test_import_load_predictor(self):
        """Test importing load_predictor from main package."""
        from dlkit import load_predictor
        assert callable(load_predictor)

    def test_import_predictor_classes(self):
        """Test importing predictor classes."""
        from dlkit.interfaces.inference import (
            CheckpointPredictor,
            IPredictor,
            PredictorConfig
        )

        assert CheckpointPredictor is not None
        assert IPredictor is not None
        assert PredictorConfig is not None

    def test_import_utilities(self):
        """Test importing utility functions."""
        from dlkit.interfaces.inference import validate_checkpoint, get_checkpoint_info

        assert callable(validate_checkpoint)
        assert callable(get_checkpoint_info)
