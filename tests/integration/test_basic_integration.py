"""Basic integration tests for dlkit workflows.

Simple end-to-end tests using fixtures from conftest.py for clean,
maintainable test dataflow management.
"""

from __future__ import annotations

import os
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

        from mlflow.tracking import MlflowClient

        from dlkit.runtime.workflows.strategies.tracking.naming import determine_experiment_name

        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_name = determine_experiment_name(mlflow_settings, mlflow_settings.MLFLOW)
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs

    def test_double_precision_training(self, double_precision_settings: GeneralSettings) -> None:
        """Regression test: double precision training must not crash on TensorDict batches.

        Lightning's DoublePrecisionPlugin.convert_input calls apply_to_collection which
        cannot reconstruct LazyStackedTensorDict from OrderedDict. This test catches
        that regression.

        Args:
            double_precision_settings: Training settings with FULL_64 precision.
        """
        result = dlkit.train(double_precision_settings)
        assert result.duration_seconds > 0

    def test_inference_basic_workflow(
        self, inference_settings: GeneralSettings, minimal_model_checkpoint: Path
    ) -> None:
        """Test basic inference workflow.

        Args:
            inference_settings: Pre-configured inference settings with dataset and checkpoint.
            minimal_model_checkpoint: Pre-created model checkpoint for inference.
        """
        # Act - Run inference using the new stateful predictor API
        # predict() mirrors model.forward() — pass tensors as positional/keyword args
        x_input = torch.randn(5, 4)

        predictor = dlkit.load_model(checkpoint_path=minimal_model_checkpoint, device="cpu")
        try:
            result = predictor.predict(x=x_input)

            # Assert - predict() returns Tensor (single output) or tuple (multi-output)
            assert result is not None
            prediction = result[0] if isinstance(result, tuple) else result
            assert isinstance(prediction, torch.Tensor)
            assert prediction.shape[0] == 5
        finally:
            predictor.unload()
