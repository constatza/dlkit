"""Inference service using CheckpointPredictor for clean train/infer separation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from dlkit.interfaces.api.domain import InferenceResult, WorkflowError
from dlkit.tools.config.workflow_configs import InferenceWorkflowConfig


class InferenceService:
    """Service for executing inference workflows via CheckpointPredictor."""

    def __init__(self) -> None:
        """Initialize inference service."""
        self.service_name = "inference_service"

    def infer(
        self,
        settings: InferenceWorkflowConfig,
        checkpoint_path: Path,
    ) -> InferenceResult:
        """Execute inference workflow using CheckpointPredictor.

        Args:
            settings: Inference workflow configuration
            checkpoint_path: Path to model checkpoint

        Returns:
            InferenceResult on success; raises WorkflowError on failure
        """
        start_time = time.time()

        try:
            from dlkit.interfaces.inference.api import load_model
            from dlkit.runtime.workflows.factories.inference_data_factory import (
                build_inference_datamodule,
            )

            predictor = load_model(checkpoint_path, auto_load=True)

            # Run batch inference if data config is present
            predictions: Any = []
            if settings.has_batch_inference_config:
                datamodule = build_inference_datamodule(settings)
                datamodule.setup("predict")
                feature_names = predictor._model_state.feature_names if predictor._model_state else ()

                for batch in datamodule.predict_dataloader():
                    features_td = batch["features"]
                    if feature_names:
                        feature_kwargs = {
                            name: features_td[name]
                            for name in feature_names
                            if name in features_td.keys()
                        }
                    else:
                        feature_kwargs = {k: features_td[k] for k in features_td.keys()}
                    output = predictor.predict(**feature_kwargs)
                    predictions.append(output)

                predictor.unload()

            duration = time.time() - start_time
            return InferenceResult(
                model_state=None,
                predictions=predictions,
                metrics=None,
                duration_seconds=duration,
            )

        except Exception as e:
            if isinstance(e, WorkflowError):
                raise
            raise WorkflowError(
                f"Inference execution failed: {str(e)}",
                {"service": self.service_name, "error": str(e)},
            )
