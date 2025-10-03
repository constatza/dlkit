"""Inference service with dual strategy support.

This module provides the main inference service that orchestrates
production and simple prediction strategies based on the type
of configuration provided.
"""

from __future__ import annotations

import time
import torch
from pathlib import Path
from typing import Any

from .config.inference_config import InferenceConfig
from .config.config_builder import build_inference_config_from_checkpoint
from .inputs.inference_input import InferenceInput
from .reconstruction import ModelReconstructionBuilder
from dlkit.tools.config import GeneralSettings
from .strategies.inference_strategy import InferenceStrategy
from .strategies.prediction_strategy import SimplePredictionStrategy
from dlkit.tools.config.workflow_settings import TrainingWorkflowSettings
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.interfaces.api.domain.errors import WorkflowError


class InferenceService:
    """Main inference service with production and prediction strategies.

    This service provides a unified interface for both inference
    (checkpoint-only) and simple prediction (Lightning-based validation).
    It automatically selects the appropriate strategy based on the inputs.
    """

    def __init__(self) -> None:
        """Initialize the inference service."""
        self._inference_strategy = InferenceStrategy()
        self._prediction_strategy = SimplePredictionStrategy()

    def infer(
        self,
        checkpoint_path: Path | str,
        inputs: InferenceInput,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True
    ) -> InferenceResult:
        """Execute inference from checkpoint only.

        This method provides standalone inference that requires only a
        model checkpoint and input data. No training configuration files
        or datasets are needed.

        Args:
            checkpoint_path: Path to trained model checkpoint
            inputs: Input data in flexible format
            device: Device specification ("auto", "cpu", "cuda", "mps")
            batch_size: Batch size for processing
            apply_transforms: Whether to apply fitted transforms

        Returns:
            InferenceResult: Inference execution result

        Raises:
            WorkflowError: On inference execution failure

        Example:
            >>> service = InferenceService()
            >>> inputs = InferenceInput({"x": torch.randn(32, 10)})
            >>> result = service.infer("model.ckpt", inputs)
            >>> predictions = result.predictions
        """
        start_time = time.time()

        try:
            # Build inference configuration from checkpoint
            config = build_inference_config_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=device,
                batch_size=batch_size,
                apply_transforms=apply_transforms
            )

            # Load model and transforms
            self._inference_strategy.load_model_from_checkpoint(config)

            # Execute inference
            result = self._inference_strategy.infer(inputs, config, batch_size)

            # Update duration
            duration = time.time() - start_time
            # Create new result with updated duration
            updated_result = InferenceResult(
                model_state=result.model_state,
                predictions=result.predictions,
                metrics=result.metrics,
                duration_seconds=duration
            )

            return updated_result

        except Exception as e:
            raise WorkflowError(
                f"Inference failed: {str(e)}",
                {"service": "inference_service", "mode": "production", "error": str(e)}
            ) from e

    def infer_with_config(
        self,
        config: InferenceConfig,
        inputs: InferenceInput,
        batch_size: int | None = None
    ) -> InferenceResult:
        """Execute inference with pre-built configuration.

        Args:
            config: Inference configuration
            inputs: Input data in flexible format
            batch_size: Batch size for processing (None = use config default)

        Returns:
            InferenceResult: Inference execution result

        Raises:
            WorkflowError: On inference execution failure
        """
        start_time = time.time()

        try:
            # Load model and transforms
            self._inference_strategy.load_model_from_checkpoint(config)

            # Execute inference
            result = self._inference_strategy.infer(inputs, config, batch_size)

            # Update duration
            duration = time.time() - start_time
            updated_result = InferenceResult(
                model_state=result.model_state,
                predictions=result.predictions,
                metrics=result.metrics,
                duration_seconds=duration
            )

            return updated_result

        except Exception as e:
            raise WorkflowError(
                f"Inference failed: {str(e)}",
                {"service": "inference_service", "mode": "production", "error": str(e)}
            ) from e

    def infer_automatic(
        self,
        checkpoint_path: Path | str,
        inputs: InferenceInput,
        device: str = "auto",
        batch_size: int = 32,
        apply_transforms: bool = True
    ) -> InferenceResult:
        """Execute inference with automatic model reconstruction from checkpoint.

        This method uses the new ModelReconstructionBuilder to automatically
        reconstruct models from enhanced checkpoint metadata without requiring
        training configuration files or manual shape parameters.

        Args:
            checkpoint_path: Path to trained model checkpoint
            inputs: Input data in flexible format
            device: Device specification ("auto", "cpu", "cuda", "mps")
            batch_size: Batch size for processing
            apply_transforms: Whether to apply fitted transforms (if available)

        Returns:
            InferenceResult: Inference execution result

        Raises:
            WorkflowError: On inference execution failure

        Example:
            >>> service = InferenceService()
            >>> inputs = InferenceInput({"x": torch.randn(32, 10)})
            >>> result = service.infer_automatic("model.ckpt", inputs)
            >>> predictions = result.predictions
        """
        start_time = time.time()

        try:
            # Reconstruct model using builder pattern
            model = (ModelReconstructionBuilder(checkpoint_path)
                    .load_checkpoint()
                    .infer_shape()
                    .reconstruct_model_settings()
                    .build_model())

            # Handle device placement
            if device == "auto":
                if hasattr(torch, "cuda") and torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            model = model.to(device)
            model.eval()

            # Execute inference directly on the reconstructed model
            with torch.no_grad():
                predictions = self._run_inference_direct(model, inputs, batch_size, apply_transforms)

            duration = time.time() - start_time
            return InferenceResult(
                model_state=None,
                predictions=predictions,
                metrics=None,
                duration_seconds=duration
            )

        except Exception as e:
            raise WorkflowError(
                f"Automatic inference failed: {str(e)}",
                {"service": "inference_service", "mode": "automatic", "error": str(e)}
            ) from e

    def _run_inference_direct(
        self,
        model: Any,
        inputs: InferenceInput,
        batch_size: int,
        apply_transforms: bool
    ) -> Any:
        """Run inference directly on a reconstructed model.

        Args:
            model: Reconstructed PyTorch Lightning model
            inputs: Input data
            batch_size: Batch size for processing
            apply_transforms: Whether to apply transforms

        Returns:
            Model predictions
        """
        import torch
        from lightning.pytorch import Trainer

        # Convert inputs to tensor format expected by model
        input_data = inputs.to_tensor_dict()

        # Create a simple dataset from inputs
        class SimpleInferenceDataset:
            def __init__(self, data_dict: dict, batch_size: int):
                self.data = data_dict
                self.batch_size = batch_size
                # Determine length from first tensor
                first_key = next(iter(data_dict.keys()))
                self.length = len(data_dict[first_key])

            def __len__(self):
                return (self.length + self.batch_size - 1) // self.batch_size

            def __iter__(self):
                for i in range(0, self.length, self.batch_size):
                    batch = {}
                    for key, tensor in self.data.items():
                        batch[key] = tensor[i:i + self.batch_size]
                    yield batch

        # Create dataset and run prediction
        dataset = SimpleInferenceDataset(input_data, batch_size)
        predictions = []

        for batch in dataset:
            # Move batch to model device
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}

            # Use model's predict_step if available, otherwise forward
            if hasattr(model, 'predict_step'):
                pred = model.predict_step(batch, 0)
            else:
                pred = model(batch)

            predictions.append(pred)

        return predictions

    def predict(
        self,
        training_settings: GeneralSettings,
        checkpoint_path: Path | str,
        **overrides
    ) -> InferenceResult:
        """Execute simple prediction using Lightning framework.

        This method uses the traditional Lightning-based inference approach
        for validation and testing scenarios where training configuration
        and datasets are available.

        Args:
            training_settings: General configuration settings
            checkpoint_path: Path to model checkpoint
            **overrides: Additional parameter overrides

        Returns:
            InferenceResult: Inference execution result

        Raises:
            WorkflowError: On inference execution failure

        Example:
            >>> from dlkit.tools.config import load_training_settings
            >>> settings = load_training_settings("config.toml")
            >>> result = service.predict(settings, "model.ckpt")
        """
        try:
            return self._prediction_strategy.predict(
                training_settings,
                checkpoint_path,
                **overrides
            )

        except Exception as e:
            raise WorkflowError(
                f"Simple prediction failed: {str(e)}",
                {"service": "inference_service", "mode": "prediction", "error": str(e)}
            ) from e

    def validate_checkpoint(
        self,
        checkpoint_path: Path | str
    ) -> dict[str, str]:
        """Validate checkpoint compatibility for inference.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        from .transforms.checkpoint_loader import CheckpointTransformLoader

        loader = CheckpointTransformLoader()
        return loader.validate_checkpoint_compatibility(checkpoint_path)

    def get_checkpoint_info(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Get information about a checkpoint file.

        Args:
            checkpoint_path: Path to model checkpoint

        Returns:
            Dictionary with checkpoint information

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        from .transforms.checkpoint_loader import CheckpointTransformLoader

        try:
            loader = CheckpointTransformLoader()

            info = {
                "checkpoint_path": str(checkpoint_path),
                "has_transforms": loader.has_transforms(checkpoint_path),
                "transform_names": loader.get_transform_names(checkpoint_path),
                "validation_errors": loader.validate_checkpoint_compatibility(checkpoint_path)
            }

            # Try to get inference metadata
            try:
                metadata = loader.get_inference_metadata(checkpoint_path)
                info["inference_metadata"] = {
                    "feature_names": metadata.get("feature_names", []),
                    "target_names": metadata.get("target_names", []),
                    "model_shape": metadata.get("model_shape"),
                }
            except Exception:
                info["inference_metadata"] = None

            return info

        except Exception as e:
            raise WorkflowError(
                f"Failed to get checkpoint info: {str(e)}",
                {"service": "inference_service", "checkpoint": str(checkpoint_path)}
            ) from e

    def is_production_ready(self) -> bool:
        """Check if production strategy is loaded and ready.

        Returns:
            True if production strategy has a loaded model
        """
        return self._inference_strategy.is_loaded()

    def unload_production_model(self) -> None:
        """Unload production model and free memory."""
        self._inference_strategy.unload_model()