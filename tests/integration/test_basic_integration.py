"""Basic integration tests for dlkit workflows.

Simple end-to-end tests using fixtures from conftest.py for clean,
maintainable test dataflow management.
"""

from __future__ import annotations

from pathlib import Path

import torch

import dlkit
import dlkit.engine.tracking.uri_resolver as uri_resolver
from dlkit.infrastructure.config.job_config import TrainingJobConfig
from dlkit.interfaces.api import train as api_train


class TestBasicIntegration:
    """Basic integration tests using conftest.py fixtures."""

    def test_vanilla_training_end_to_end(self, training_settings: TrainingJobConfig) -> None:
        """Test basic vanilla training workflow end-to-end.

        Args:
            training_settings: Pre-configured settings with minimal dataset.
        """
        training_result = api_train(training_settings)

        assert training_result.duration_seconds > 0
        assert training_result.metrics is not None

    def test_mlflow_training_basic(self, mlflow_settings: TrainingJobConfig) -> None:
        """Test basic MLflow training workflow.

        Args:
            mlflow_settings: Pre-configured settings with MLflow enabled.
        """
        training_result = api_train(mlflow_settings)

        assert training_result.duration_seconds > 0
        if training_result.metrics:
            assert isinstance(training_result.metrics, dict)

        from mlflow.tracking import MlflowClient

        from dlkit.engine.tracking.naming import determine_experiment_name

        tracking_uri = training_result.mlflow_tracking_uri
        assert tracking_uri is not None
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_name = determine_experiment_name(mlflow_settings)
        experiment = client.get_experiment_by_name(experiment_name)
        assert experiment is not None

        runs = client.search_runs(
            [experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        assert runs

    def test_double_precision_training(self, double_precision_settings: TrainingJobConfig) -> None:
        """Regression test: double precision training must not crash on TensorDict batches.

        Lightning's DoublePrecisionPlugin.convert_input calls apply_to_collection which
        cannot reconstruct LazyStackedTensorDict from OrderedDict. This test catches
        that regression.

        Args:
            double_precision_settings: Training settings with FULL_64 precision.
        """
        result = api_train(double_precision_settings)
        assert result.duration_seconds > 0

    def test_vanilla_training_keeps_outputs_local_without_mlflow(
        self,
        training_settings: TrainingJobConfig,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """Vanilla training should avoid MLflow state and keep local outputs deterministic."""
        monkeypatch.delenv("MLFLOW_ARTIFACT_URI", raising=False)
        monkeypatch.setattr(uri_resolver, "local_host_alive", lambda: False)

        training_cfg = training_settings.training
        assert training_cfg is not None
        trainer_cfg = training_cfg.trainer
        assert trainer_cfg is not None

        output_root = tmp_path / "local_training_output"
        settings = training_settings.model_copy(
            update={
                "training": training_cfg.model_copy(
                    update={
                        "trainer": trainer_cfg.model_copy(
                            update={
                                "enable_checkpointing": True,
                                "default_root_dir": output_root,
                                "fast_dev_run": False,
                            }
                        )
                    }
                )
            }
        )

        training_result = api_train(settings)

        assert training_result.duration_seconds > 0
        assert training_result.mlflow_run_id is None
        assert training_result.mlflow_tracking_uri is None
        assert "mlflow_run_id" not in training_result.metrics
        assert "mlflow_tracking_uri" not in training_result.metrics

        checkpoint_path = training_result.checkpoint_path
        assert checkpoint_path is not None
        assert checkpoint_path.parent == output_root / "checkpoints"
        assert checkpoint_path.parent.exists()
        assert checkpoint_path.is_relative_to(output_root)

        assert not (tmp_path / "mlruns").exists()
        assert not (tmp_path / "mlartifacts").exists()

    def test_inference_basic_workflow(
        self, inference_settings: TrainingJobConfig, minimal_model_checkpoint: Path
    ) -> None:
        """Test basic inference workflow.

        Args:
            inference_settings: Pre-configured inference settings with dataset and checkpoint.
            minimal_model_checkpoint: Pre-created model checkpoint for inference.
        """
        x_input = torch.randn(5, 4)

        predictor = dlkit.load_model(checkpoint_path=minimal_model_checkpoint, device="cpu")
        try:
            result = predictor.predict(x=x_input)

            assert result is not None
            prediction = result.predictions
            assert isinstance(prediction, torch.Tensor)
            assert prediction.shape[0] == 5
        finally:
            predictor.unload()
