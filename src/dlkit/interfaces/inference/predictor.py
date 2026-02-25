"""Stateful predictor for efficient inference.

Consolidated predictor without hexagonal architecture overhead.
All loading logic integrated directly - no use case objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

import torch
from loguru import logger

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config.precision.strategy import PrecisionStrategy

from .config import PredictorConfig, ModelState, InferenceResult
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
    ) -> InferenceResult:
        """Execute inference on loaded model."""
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

    def _name_predictions(self, predictions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Name single tensor predictions based on target configuration.

        Args:
            predictions: Raw model output tensor

        Returns:
            Dict with properly named predictions (e.g., {"y": tensor})
        """
        # Guard clause: If model_state not available, use default name
        if self._model_state is None:
            return {"output": predictions}

        # Get entry configs from checkpoint metadata
        # Need type guard since metadata values can be various types
        inference_metadata_raw = self._model_state.metadata.get("inference_metadata", {})
        if not isinstance(inference_metadata_raw, dict):
            return {"output": predictions}

        entry_configs_raw = inference_metadata_raw.get("entry_configs", {})
        if not isinstance(entry_configs_raw, dict):
            return {"output": predictions}

        entry_configs = entry_configs_raw

        # Guard clause: No entry configs, use default name
        if not entry_configs:
            return {"output": predictions}

        # Find target entries
        target_names = [
            name for name, config in entry_configs.items()
            if config.get("type") == "target"
        ]

        # Use match-case for cleaner logic (avoiding nested ifs)
        match len(target_names):
            case 0:
                # No targets found, use default
                return {"output": predictions}
            case 1:
                # Single target - use its name
                return {target_names[0]: predictions}
            case _:
                # Multiple targets - use first one
                # (This is ambiguous but better than failing)
                logger.warning(
                    f"Multiple targets found: {target_names}. "
                    f"Using first target '{target_names[0]}' for naming."
                )
                return {target_names[0]: predictions}

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
    ) -> InferenceResult:
        """Execute inference on loaded model.

        Args:
            inputs: Input data (dict or tensor)
            batch_size: Optional batch size override

        Returns:
            InferenceResult with predictions and model state

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
            metadata = self._model_state.metadata
            _feature_names_raw = metadata.get("feature_names", [])
            feature_names: list[str] = (
                _feature_names_raw if isinstance(_feature_names_raw, list) else []
            )

            with torch.no_grad():
                if len(inputs) == 1:
                    predictions = self._model_state.model(next(iter(inputs.values())))
                elif feature_names:
                    tensors = tuple(inputs[k] for k in feature_names if k in inputs)
                    predictions = self._model_state.model(*tensors)
                else:
                    # Fallback: insertion order (Python 3.7+ dict ordering guarantee)
                    predictions = self._model_state.model(*inputs.values())

            # Apply inverse target transforms if requested
            if self._config.apply_transforms and self._model_state.target_transforms:
                logger.debug("Applying inverse target transforms")
                predictions = apply_inverse_transforms(
                    predictions,
                    self._model_state.target_transforms
                )

            # Wrap single tensor predictions in dict with proper name
            if isinstance(predictions, torch.Tensor):
                predictions = self._name_predictions(predictions)

            # Return wrapped in InferenceResult dataclass (better than bare tensor)
            return InferenceResult(predictions=predictions)

    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready.

        Returns:
            True if model is loaded
        """
        return self._loaded and self._model_state is not None

    @property
    def model(self) -> torch.nn.Module | None:
        """Access the underlying PyTorch model.

        Useful for:
        - Custom analysis or inspection
        - Extracting intermediate layer activations
        - Fine-tuning or further training
        - Direct model manipulation

        Returns:
            The loaded PyTorch model, or None if not loaded

        Example:
            >>> predictor = load_model(checkpoint_path)
            >>> model = predictor.model
            >>> # Access model layers, parameters, etc.
        """
        return self._model_state.model if self._model_state is not None else None

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
