"""Integration tests for the new predictor-based inference API.

NOTE: Core predictor functionality is comprehensively tested in
tests/interfaces/inference/test_simplified_predictor.py (18 tests).

This file focuses on API imports and backward compatibility documentation.
"""

from __future__ import annotations

import pytest


class TestPredictorAPIImports:
    """Test that new predictor API can be imported correctly."""

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


class TestBackwardCompatibilityNote:
    """Document breaking changes and migration path."""

    def test_old_infer_api_removed(self):
        """OLD API (removed): infer() function."""
        # This test documents the old API that was removed

        # OLD (no longer works):
        # from dlkit import infer
        # result = infer("model.ckpt", inputs)

        # NEW (use this instead):
        # from dlkit import load_predictor
        # with load_predictor("model.ckpt") as predictor:
        #     result = predictor.predict(inputs)

        # Or for multiple predictions:
        # predictor = load_predictor("model.ckpt")
        # result1 = predictor.predict(input1)
        # result2 = predictor.predict(input2)
        # predictor.unload()

        # Verify old API doesn't exist
        with pytest.raises(ImportError):
            from dlkit import infer  # Should fail
