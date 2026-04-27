"""Stateful predictor for efficient inference.

Consolidated predictor without hexagonal architecture overhead.
All loading logic integrated directly - no use case objects.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable

import torch
from tensordict import TensorDict

from dlkit.common.errors import WorkflowError
from dlkit.engine.adapters.lightning.base import _unpack_model_output
from dlkit.infrastructure.precision import (
    PrecisionService,
    get_precision_service,
    precision_override,
)
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from dlkit.infrastructure.utils.logging_config import get_logger

from .config import ModelState, PredictorConfig
from .loading import build_model_from_checkpoint, load_checkpoint
from .shapes import infer_shape_specification
from .transforms import load_transforms_from_checkpoint

logger = get_logger(__name__)


class PredictorError(WorkflowError):
    """Base exception for predictor-related errors."""


class PredictorNotLoadedError(PredictorError):
    """Raised when attempting operations on unloaded predictor."""

    def __init__(
        self, message: str = "Predictor not loaded. Call load() first or use context manager."
    ):
        super().__init__(message, {"error_type": "PredictorNotLoadedError"})


@runtime_checkable
class IPredictor(Protocol):
    """Protocol for stateful inference predictors.

    Predictors encapsulate a loaded model and provide efficient
    inference across multiple calls without reloading from checkpoint.

    Industry-standard pattern following PyTorch, scikit-learn, Hugging Face.
    """

    def predict(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | TensorDict | tuple[Any, ...]:
        """Execute inference, mirroring model.forward() signature."""
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
        >>> result1 = predictor.predict(x_tensor)
        >>> result2 = predictor.predict(x=x_tensor, edge_attr=ea_tensor)
        >>>
        >>> # Clean up
        >>> predictor.unload()

        >>> # Or with context manager
        >>> with CheckpointPredictor(config, auto_load=True) as predictor:
        ...     result = predictor.predict(x=inputs)
    """

    def __init__(
        self,
        config: PredictorConfig,
        precision_service: PrecisionService | None = None,
    ) -> None:
        """Initialize predictor with configuration.

        Args:
            config: Predictor configuration
            precision_service: Optional precision service override. Defaults to the global
                precision service when not provided.
        """
        self._config = config
        self._precision_service = (
            precision_service if precision_service is not None else get_precision_service()
        )

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
            logger.debug("Predictor already loaded")
            return self

        logger.info(f"Loading predictor from {self._config.checkpoint_path}")

        # Load checkpoint
        checkpoint = load_checkpoint(self._config.checkpoint_path)

        # Infer shape specification
        logger.debug("Inferring shape specification")
        shape_spec = infer_shape_specification(checkpoint, dataset=None)

        # Build and load model
        logger.debug("Building model from checkpoint")
        model = build_model_from_checkpoint(checkpoint, shape_spec)

        # Load transforms (separated by type)
        logger.debug("Loading fitted transforms")
        feature_transforms, target_transforms = load_transforms_from_checkpoint(checkpoint)

        # Place model on device
        device = self._resolve_device()
        logger.debug("Moving model to device: {}", device)
        model = model.to(device)

        # Ensure eval mode
        model.eval()

        # Extract feature_names and predict_target_key from checkpoint metadata
        meta = checkpoint.get("dlkit_metadata", {})
        raw_fn = meta.get("feature_names", ())
        feature_names: tuple[str, ...] = tuple(raw_fn) if isinstance(raw_fn, (list, tuple)) else ()
        predict_target_key: str = str(meta.get("predict_target_key", ""))

        # Create model state
        self._model_state = ModelState(
            model=model,
            device=device,
            shape_spec=shape_spec,
            feature_transforms=feature_transforms if feature_transforms else None,
            target_transforms=target_transforms if target_transforms else None,
            metadata=meta,
            feature_names=feature_names,
            predict_target_key=predict_target_key,
        )

        # Infer precision from model
        self._inferred_precision = self._precision_service.infer_precision_from_model(model)

        self._loaded = True
        logger.info(
            "Predictor loaded: checkpoint='{}' device='{}' precision='{}'",
            self._config.checkpoint_path,
            device,
            self._inferred_precision,
        )

        return self

    def predict(
        self,
        *args: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> torch.Tensor | TensorDict | tuple[Any, ...]:
        """Execute inference, mirroring model.forward() signature.

        Applies feature transforms to inputs (by position for args, by name for
        kwargs), calls model.forward(*args, **kwargs), then applies inverse target
        transform to the first output element.

        Args:
            *args: Positional feature tensors. Each args[i] is transformed using
                the transform for ``feature_names[i]`` (from checkpoint metadata).
            **kwargs: Named feature tensors. Each kwarg is transformed using the
                transform registered under the same key name.

        Returns:
            Single Tensor or TensorDict for single-output models.
            tuple for multi-output (first element is the prediction after
            inverse transform; remaining are latents/auxiliary — returned as-is).

        Raises:
            PredictorNotLoadedError: If predictor not loaded.
        """
        if not self._loaded or self._model_state is None:
            raise PredictorNotLoadedError()

        # Establish precision context (inferred from model)
        precision_to_use = self._config.precision or self._inferred_precision

        if precision_to_use is not None:
            ctx = precision_override(precision_to_use)
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            if self._config.apply_transforms:
                args, kwargs = self._apply_input_transforms(args, kwargs)

            with torch.no_grad():
                raw_output = self._model_state.model(*args, **kwargs)

            predictions_raw, latents_raw = _unpack_model_output(raw_output)

            if self._config.apply_transforms:
                predictions_raw = self._apply_output_inverse_transform(predictions_raw)

            if latents_raw is None:
                return predictions_raw

            # Unpack latents tuple for clean API: (pred, lat0, lat1, ...)
            if isinstance(latents_raw, tuple):
                return (predictions_raw, *latents_raw)
            return (predictions_raw, latents_raw)

    def _apply_input_transforms(
        self,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        """Apply forward transforms to positional and keyword inputs.

        Positional arg ``i`` is transformed using the chain registered under
        ``feature_names[i]``.  Keyword arg ``k`` is transformed using the
        chain registered under ``k``.

        Args:
            args: Positional input tensors.
            kwargs: Named input tensors.

        Returns:
            Transformed (args, kwargs) tuple.
        """
        if self._model_state is None:
            return args, kwargs
        ft = self._model_state.feature_transforms
        if not ft:
            return args, kwargs

        fn = self._model_state.feature_names
        transformed_args = tuple(
            ft[fn[i]](t) if i < len(fn) and fn[i] in ft else t for i, t in enumerate(args)
        )
        transformed_kwargs: dict[str, torch.Tensor] = {
            k: ft[k](t) if k in ft else t for k, t in kwargs.items()
        }
        return transformed_args, transformed_kwargs

    def _apply_output_inverse_transform(
        self,
        predictions: torch.Tensor | TensorDict,
    ) -> torch.Tensor | TensorDict:
        """Apply inverse target transform to the primary model output.

        Uses ``predict_target_key`` to select the transform chain from
        ``target_transforms``.  Skips gracefully when no key is configured,
        the key is absent, or the chain is not invertible.

        Args:
            predictions: Primary model output (Tensor or TensorDict).

        Returns:
            Inverse-transformed predictions (same type as input).
        """
        if self._model_state is None:
            return predictions
        tt = self._model_state.target_transforms
        target_key = self._model_state.predict_target_key
        if not tt or not target_key or target_key not in tt:
            return predictions

        from dlkit.domain.transforms.base import InvertibleTransform

        chain = tt[target_key]
        if not isinstance(chain, InvertibleTransform):
            return predictions

        if isinstance(predictions, torch.Tensor):
            return chain.inverse_transform(predictions)

        # TensorDict: apply inverse to its "predictions" leaf if present
        if isinstance(predictions, TensorDict) and "predictions" in predictions.keys():
            from typing import cast as _cast

            return _cast(
                TensorDict,
                predictions.apply(
                    lambda t: chain.inverse_transform(t), batch_size=predictions.batch_size
                ),
            )

        return predictions

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
                logger.debug("Moving model to CPU before unload")
                self._model_state.model.cpu()

            self._model_state = None

        self._loaded = False
        self._inferred_precision = None

        # Force garbage collection
        import gc

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("Predictor unloaded")

    def _resolve_device(self) -> str:
        """Resolve device specification to actual device string.

        Returns:
            Device string (e.g., "cpu", "cuda", "mps")
        """
        device_spec = self._config.device

        if device_spec == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
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
