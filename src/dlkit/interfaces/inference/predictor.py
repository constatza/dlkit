"""Stateful predictor for efficient inference.

Consolidated predictor without hexagonal architecture overhead.
All loading logic integrated directly - no use case objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, Self

import torch
from loguru import logger

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.precision.strategy import PrecisionStrategy

from .config import PredictorConfig, ModelState
from .loading import load_checkpoint, build_model_from_checkpoint
from .shapes import infer_shape_specification
from .transforms import (
    load_transforms_from_checkpoint,
    apply_transforms,
    apply_inverse_transforms
)


class PredictorError(WorkflowError):
    """Base exception for predictor-related errors."""
    pass


class PredictorNotLoadedError(PredictorError):
    """Raised when attempting operations on unloaded predictor."""

    def __init__(self, message: str = "Predictor not loaded. Call load() first or use context manager."):
        super().__init__(message, {"error_type": "PredictorNotLoadedError"})


class IPredictor(Protocol):
    """Protocol for stateful inference predictors.

    Predictors encapsulate a loaded model and provide efficient
    inference across multiple calls without reloading from checkpoint.

    Industry-standard pattern following PyTorch, scikit-learn, Hugging Face.
    """

    def predict(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        batch_size: int | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute inference on loaded model."""
        ...

    def predict_from_config(
        self,
        config: GeneralSettings | Path | str
    ) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        """Execute batch inference using dataset from config."""
        ...

    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready."""
        ...

    def unload(self) -> None:
        """Unload model and free resources."""
        ...


class CheckpointPredictor(IPredictor):
    """Stateful predictor that loads model once and reuses it.

    Consolidated implementation - all loading logic integrated directly.
    No use case objects, no hexagonal architecture overhead.

    Usage:
        >>> # Load once
        >>> predictor = CheckpointPredictor(config)
        >>> predictor.load()
        >>>
        >>> # Predict many times (no reloading!)
        >>> result1 = predictor.predict(input1)
        >>> result2 = predictor.predict(input2)
        >>>
        >>> # Clean up
        >>> predictor.unload()

        >>> # Or with context manager
        >>> with CheckpointPredictor(config, auto_load=True) as predictor:
        ...     result = predictor.predict(inputs)
    """

    def __init__(self, config: PredictorConfig):
        """Initialize predictor with configuration.

        Args:
            config: Predictor configuration
        """
        self._config = config
        self._precision_service = get_precision_service()

        # State
        self._model_state: ModelState | None = None
        self._loaded = False
        self._inferred_precision: PrecisionStrategy | None = None

        # Auto-load if requested
        if config.auto_load:
            self.load()

    def load(self) -> Self:
        """Load model from checkpoint (expensive - call once).

        Consolidates all loading logic:
        - Checkpoint loading
        - Shape inference
        - Model reconstruction
        - Transform loading
        - Precision inference
        - Device placement

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If loading fails
        """
        if self._loaded:
            logger.info("Predictor already loaded")
            return self

        logger.info(f"Loading predictor from {self._config.checkpoint_path}")

        # Load checkpoint
        checkpoint = load_checkpoint(self._config.checkpoint_path)

        # Infer shape specification
        logger.info("Inferring shape specification")
        shape_spec = infer_shape_specification(checkpoint, dataset=None)

        # Build and load model
        logger.info("Building model from checkpoint")
        model = build_model_from_checkpoint(checkpoint, shape_spec)

        # Load transforms (separated by type)
        logger.info("Loading fitted transforms")
        feature_transforms, target_transforms = load_transforms_from_checkpoint(checkpoint)

        # Place model on device
        device = self._resolve_device()
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)

        # Ensure eval mode
        model.eval()

        # Create model state
        self._model_state = ModelState(
            model=model,
            device=device,
            shape_spec=shape_spec,
            feature_transforms=feature_transforms if feature_transforms else None,
            target_transforms=target_transforms if target_transforms else None,
            metadata=checkpoint.get("dlkit_metadata", {})
        )

        # Infer precision from model
        self._inferred_precision = self._precision_service.infer_precision_from_model(model)
        logger.info(f"Inferred precision from model: {self._inferred_precision}")

        self._loaded = True
        logger.info("Predictor loaded successfully")

        return self

    def predict(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        batch_size: int | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Execute inference on loaded model.

        Args:
            inputs: Input data (dict or tensor)
            batch_size: Optional batch size override

        Returns:
            Predictions tensor or dict

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If inference fails
        """
        if not self._loaded or self._model_state is None:
            raise PredictorNotLoadedError()

        # Establish precision context (inferred from model)
        precision_to_use = self._config.precision or self._inferred_precision

        # Only use precision override if we have a precision value
        if precision_to_use is not None:
            ctx = precision_override(precision_to_use)
        else:
            # No-op context manager
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            # Convert inputs to dict format if needed
            if not isinstance(inputs, dict):
                inputs = {"x": inputs}

            # Apply feature transforms if requested
            if self._config.apply_transforms and self._model_state.feature_transforms:
                logger.debug("Applying feature transforms")
                inputs = apply_transforms(inputs, self._model_state.feature_transforms)

            # Model forward pass with no_grad
            with torch.no_grad():
                # Extract first value if single-entry dict
                if len(inputs) == 1:
                    model_input = next(iter(inputs.values()))
                else:
                    model_input = inputs

                predictions = self._model_state.model(model_input)

            # Apply inverse target transforms if requested
            if self._config.apply_transforms and self._model_state.target_transforms:
                logger.debug("Applying inverse target transforms")
                predictions = apply_inverse_transforms(
                    predictions,
                    self._model_state.target_transforms
                )

            # Return predictions directly (simpler than wrapping in InferenceResult)
            return predictions

    def predict_from_config(
        self,
        config: GeneralSettings | Path | str
    ) -> Iterator[torch.Tensor | dict[str, torch.Tensor]]:
        """Execute batch inference using dataset from config.

        Args:
            config: Configuration with dataset settings

        Yields:
            Predictions for each batch

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If config loading fails
        """
        if not self._loaded:
            raise PredictorNotLoadedError()

        # TODO: Implement config-based batch inference
        # This requires proper datamodule loading which is complex
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "predict_from_config not yet implemented in simplified architecture. "
            "Use predict() directly with a dataloader."
        )

    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready.

        Returns:
            True if model is loaded
        """
        return self._loaded and self._model_state is not None

    def unload(self) -> None:
        """Unload model and free resources."""
        if self._model_state is not None:
            # Move to CPU and delete to free GPU memory
            if self._model_state.device != "cpu":
                logger.info("Moving model to CPU before unload")
                self._model_state.model.cpu()

            self._model_state = None

        self._loaded = False
        self._inferred_precision = None

        # Force garbage collection
        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Predictor unloaded")

    def _resolve_device(self) -> str:
        """Resolve device specification to actual device string.

        Returns:
            Device string (e.g., "cpu", "cuda", "mps")
        """
        device_spec = self._config.device

        if device_spec == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return device_spec

    # Context manager support
    def __enter__(self) -> Self:
        """Enter context manager."""
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        self.unload()
