"""Stateful predictor abstraction for efficient inference.

This module provides the core predictor abstraction following industry-standard
patterns (PyTorch, scikit-learn, Hugging Face). The predictor encapsulates
a loaded model and enables efficient reuse across multiple predictions without
reloading the checkpoint.

Design Pattern: State + Context Manager
- Load model once in __init__ or load()
- Reuse for multiple predict() calls
- Clean up with unload() or context manager
"""

from __future__ import annotations

import time
import torch
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterator, Protocol, Self
from dataclasses import dataclass
from loguru import logger

from dlkit.interfaces.api.domain.errors import WorkflowError
from dlkit.interfaces.api.domain.models import InferenceResult
from dlkit.interfaces.api.domain.precision import precision_override
from dlkit.interfaces.api.services.precision_service import get_precision_service
from dlkit.tools.config import GeneralSettings
from dlkit.tools.config.precision.strategy import PrecisionStrategy

from .domain.models import ModelState, InferenceRequest
from .inputs.inference_input import InferenceInput


class PredictorError(WorkflowError):
    """Base exception for predictor-related errors."""
    pass


class PredictorNotLoadedError(PredictorError):
    """Raised when attempting operations on unloaded predictor."""

    def __init__(self, message: str = "Predictor not loaded. Call load() first or use context manager."):
        super().__init__(message, {"error_type": "PredictorNotLoadedError"})


class PredictorStateError(PredictorError):
    """Raised when predictor is in invalid state for operation."""
    pass


class IPredictor(Protocol):
    """Protocol for stateful inference predictors.

    Predictors encapsulate a loaded model and provide efficient
    inference across multiple calls without reloading from checkpoint.

    This follows industry-standard patterns:
    - PyTorch: model = load(); model.eval(); output = model(x)
    - scikit-learn: estimator.fit(); estimator.predict()
    - Hugging Face: pipeline = pipeline(...); output = pipeline(x)

    Usage:
        >>> # Load once
        >>> predictor = load_predictor("model.ckpt", device="cuda")
        >>>
        >>> # Predict many times (no reloading!)
        >>> result1 = predictor.predict(input1)
        >>> result2 = predictor.predict(input2)
        >>> result3 = predictor.predict(input3)
        >>>
        >>> # Clean up
        >>> predictor.unload()

        >>> # Or with context manager
        >>> with load_predictor("model.ckpt") as predictor:
        ...     result = predictor.predict(inputs)
    """

    @abstractmethod
    def predict(
        self,
        inputs: InferenceInput | dict[str, Any] | Any,
        batch_size: int | None = None
    ) -> InferenceResult:
        """Execute inference on loaded model (fast operation).

        This method performs ONLY the forward pass - no checkpoint loading,
        no model reconstruction. It's designed to be called hundreds of times
        efficiently.

        Args:
            inputs: Input data for inference (flexible formats supported)
            batch_size: Optional batch size override

        Returns:
            InferenceResult with predictions

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If inference fails
        """
        ...

    @abstractmethod
    def predict_from_config(
        self,
        config: GeneralSettings | Path | str
    ) -> Iterator[InferenceResult]:
        """Execute batch inference using dataset/split from config.

        This method enables inference on full datasets defined in configuration
        files. It builds a dataloader from the config's dataset/split settings
        and yields results per batch.

        Args:
            config: Configuration with dataset/split/datamodule settings

        Yields:
            InferenceResult for each batch

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If config loading or inference fails

        Example:
            >>> predictor = load_predictor("model.ckpt")
            >>> for result in predictor.predict_from_config("inference_config.toml"):
            ...     process_batch(result)
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready for inference.

        Returns:
            True if model is loaded and ready
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free GPU/CPU resources.

        After calling this, the predictor is no longer usable.
        Call load() again to re-initialize.
        """
        ...


@dataclass
class PredictorConfig:
    """Configuration for predictor creation and initialization.

    Encapsulates all parameters needed to create and configure a predictor.
    """
    checkpoint_path: Path
    device: str = "auto"
    batch_size: int = 32
    apply_transforms: bool = True
    auto_load: bool = True
    precision: PrecisionStrategy | None = None  # Optional precision override


class CheckpointPredictor(IPredictor):
    """Stateful predictor that loads model from checkpoint once and reuses it.

    This is the main inference object for production use. It encapsulates:
    - Loaded PyTorch model in eval mode
    - Transform executor (if applicable)
    - Inference configuration
    - Device placement

    The predictor follows explicit lifecycle management:
    1. Construction: Create predictor (optionally auto-load)
    2. Loading: Load model from checkpoint (expensive, done once)
    3. Prediction: Execute inference (fast, done many times)
    4. Cleanup: Unload model and free resources

    Usage:
        >>> # Explicit loading
        >>> predictor = CheckpointPredictor(...)
        >>> predictor.load()
        >>> result = predictor.predict(inputs)
        >>> predictor.unload()

        >>> # Context manager (recommended)
        >>> with CheckpointPredictor(..., auto_load=True) as predictor:
        ...     result = predictor.predict(inputs)
    """

    def __init__(
        self,
        config: PredictorConfig,
        model_loading_use_case: Any,  # ModelLoadingUseCase
        inference_execution_use_case: Any,  # InferenceExecutionUseCase
    ):
        """Initialize predictor with configuration and dependencies.

        Args:
            config: Predictor configuration
            model_loading_use_case: Use case for loading models
            inference_execution_use_case: Use case for executing inference
        """
        self._config = config
        self._model_loading_use_case = model_loading_use_case
        self._inference_execution_use_case = inference_execution_use_case
        self._precision_service = get_precision_service()

        # State
        self._model_state: ModelState | None = None
        self._loaded = False
        self._inferred_precision: PrecisionStrategy | None = None

        # Auto-load if requested
        if config.auto_load:
            self.load()

    def load(self) -> Self:
        """Load model from checkpoint (expensive operation - call once).

        This method performs all expensive operations:
        - Checkpoint loading from disk
        - Model reconstruction
        - Weight loading
        - Device placement
        - Transform loading
        - Eval mode setup
        - Precision inference from model dtype

        Returns:
            Self for method chaining

        Raises:
            WorkflowError: If loading fails
            PredictorStateError: If already loaded
        """
        if self._loaded:
            # Already loaded, no-op
            return self

        # Load model using the loading use case
        self._model_state = self._model_loading_use_case.load_model(
            checkpoint_path=self._config.checkpoint_path,
            device=self._config.device
        )

        # Determine precision: use configured precision if provided, otherwise infer from model
        if self._config.precision is not None:
            # Use explicitly configured precision
            self._inferred_precision = self._config.precision
            logger.info(f"Using configured precision: {self._inferred_precision}")
        elif self._model_state and self._model_state.model:
            # Infer precision from loaded model for data loading consistency
            self._inferred_precision = self._precision_service.infer_precision_from_model(
                self._model_state.model
            )
            if self._inferred_precision:
                logger.info(f"Inferred precision from model: {self._inferred_precision}")
            else:
                logger.warning("Could not infer precision from model, using default float32")

        self._loaded = True
        return self

    def predict(
        self,
        inputs: InferenceInput | dict[str, Any] | Any,
        batch_size: int | None = None
    ) -> InferenceResult:
        """Execute inference on loaded model (fast operation).

        Args:
            inputs: Input data (flexible formats)
            batch_size: Optional batch size override

        Returns:
            InferenceResult with predictions

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If inference fails
        """
        if not self._loaded or self._model_state is None:
            raise PredictorNotLoadedError()

        # Convert inputs to InferenceInput if needed
        if not isinstance(inputs, InferenceInput):
            inputs = InferenceInput(inputs)

        # Use batch size from config if not overridden
        effective_batch_size = batch_size if batch_size is not None else self._config.batch_size

        # Determine dtype for tensor conversion based on inferred precision
        target_dtype = torch.float32  # Default
        if self._inferred_precision is not None:
            target_dtype = self._inferred_precision.to_torch_dtype()

        logger.debug(f"Converting inputs to dtype: {target_dtype}")

        # Create inference request
        request = InferenceRequest(
            inputs=inputs.to_tensor_dict(dtype=target_dtype),
            batch_size=effective_batch_size,
            apply_transforms=self._config.apply_transforms,
            device=self._config.device
        )

        # Apply inferred precision context if available
        if self._inferred_precision is not None:
            with precision_override(self._inferred_precision):
                return self._inference_execution_use_case.execute_inference(
                    model_state=self._model_state,
                    request=request
                )
        else:
            # No precision inferred, use default behavior
            return self._inference_execution_use_case.execute_inference(
                model_state=self._model_state,
                request=request
            )

    def predict_from_config(
        self,
        config: GeneralSettings | Path | str
    ) -> Iterator[InferenceResult]:
        """Execute batch inference using dataset from config.

        This method builds a dataloader from the configuration's dataset/split
        settings and iterates through batches, yielding results.

        Args:
            config: Configuration path or loaded settings

        Yields:
            InferenceResult for each batch

        Raises:
            PredictorNotLoadedError: If predictor not loaded
            WorkflowError: If config loading or inference fails
        """
        if not self._loaded or self._model_state is None:
            raise PredictorNotLoadedError()

        # Load config if path provided
        if isinstance(config, (Path, str)):
            from dlkit.tools.config import load_settings
            config = load_settings(Path(config))

        # Apply precision context BEFORE building dataloader
        # This ensures data loading happens with correct dtype
        if self._inferred_precision is not None:
            context = precision_override(self._inferred_precision)
            context.__enter__()
        else:
            context = None

        # Build dataloader from config (within precision context)
        from .application.config_inference import build_prediction_dataloader_from_config
        dataloader = build_prediction_dataloader_from_config(
            config=config,
            model_state=self._model_state,
            device=self._config.device
        )

        try:
            # Determine dtype for tensor conversion based on inferred precision
            target_dtype = torch.float32  # Default
            if self._inferred_precision is not None:
                target_dtype = self._inferred_precision.to_torch_dtype()

            logger.debug(f"Using dtype for config-based inference: {target_dtype}")

            # Iterate and predict on each batch
            for batch_idx, batch in enumerate(dataloader):
                # Extract inputs from batch
                if isinstance(batch, dict):
                    inputs = batch
                elif isinstance(batch, (list, tuple)):
                    # Handle (features, targets) tuple
                    inputs = batch[0] if len(batch) > 0 else batch
                else:
                    inputs = batch

                # Convert to InferenceInput
                inference_input = InferenceInput(inputs)

                # Create request with correct dtype
                request = InferenceRequest(
                    inputs=inference_input.to_tensor_dict(dtype=target_dtype),
                    batch_size=len(inputs) if hasattr(inputs, '__len__') else 1,
                    apply_transforms=False,  # Transforms already applied by dataloader
                    device=self._config.device,
                    metadata={"batch_idx": batch_idx}
                )

                # Execute inference
                result = self._inference_execution_use_case.execute_inference(
                    model_state=self._model_state,
                    request=request
                )

                yield result
        finally:
            # Clean up precision context
            if context is not None:
                context.__exit__(None, None, None)

    def is_loaded(self) -> bool:
        """Check if predictor is loaded and ready.

        Returns:
            True if model is loaded
        """
        return self._loaded and self._model_state is not None

    def unload(self) -> None:
        """Unload model and free resources.

        This method frees GPU/CPU memory by removing model references
        and clearing CUDA cache if applicable.
        """
        if self._loaded:
            self._model_state = None
            self._loaded = False

            # Free GPU memory if CUDA is available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_config(self) -> PredictorConfig:
        """Get predictor configuration.

        Returns:
            PredictorConfig used to create this predictor
        """
        return self._config

    def get_checkpoint_path(self) -> Path:
        """Get the checkpoint path used by this predictor.

        Returns:
            Path to model checkpoint
        """
        return self._config.checkpoint_path

    # Context manager support

    def __enter__(self) -> Self:
        """Context manager entry - ensure model is loaded.

        Returns:
            Self (loaded)
        """
        if not self._loaded:
            self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically unload model.

        Returns:
            False to propagate exceptions
        """
        self.unload()
        return False
