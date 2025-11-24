"""Basic integration tests for dlkit workflows.

Simple end-to-end tests using fixtures from conftest.py for clean,
maintainable test dataflow management.
"""

from __future__ import annotations

from pathlib import Path

import torch
import dlkit
from dlkit.tools.config import GeneralSettings


class TestBasicIntegration:
    """Basic integration tests using conftest.py fixtures."""

    def test_vanilla_training_end_to_end(self, training_settings: GeneralSettings) -> None:
        """Test basic vanilla training workflow end-to-end.

        Args:
            training_settings: Pre-configured settings with minimal dataset.
        """
        # Act - Run training
        training_result = dlkit.train(training_settings)

        # Assert - Basic validation
        assert training_result.duration_seconds > 0
        assert training_result.metrics is not None

    def test_mlflow_training_basic(self, mlflow_settings: GeneralSettings) -> None:
        """Test basic MLflow training workflow.

        Args:
            mlflow_settings: Pre-configured settings with MLflow enabled.
        """
        # Act - Run MLflow training
        training_result = dlkit.train(mlflow_settings)

        # Assert - Validate results
        assert training_result.duration_seconds > 0
        if training_result.metrics:
            assert isinstance(training_result.metrics, dict)

    def test_inference_basic_workflow(
        self, inference_settings: GeneralSettings, minimal_model_checkpoint: Path
    ) -> None:
        """Test basic inference workflow.

        Args:
            inference_settings: Pre-configured inference settings with dataset and checkpoint.
            minimal_model_checkpoint: Pre-created model checkpoint for inference.
        """
        # Act - Run inference using the new stateful predictor API
        # Convert settings to inputs
        test_inputs = {"x": torch.randn(5, 4), "y": torch.randn(5, 1)}  # Basic test inputs

        # Use the new predictor API
        predictor = dlkit.load_predictor(checkpoint_path=minimal_model_checkpoint, device="cpu")
        try:
            predictions = predictor.predict(test_inputs)

            # Assert - Validate results
            assert predictions is not None
        finally:
            predictor.unload()
