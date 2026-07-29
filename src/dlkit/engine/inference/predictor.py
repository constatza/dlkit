"""Stateful predictor for efficient inference.

Consolidated predictor without hexagonal architecture overhead.
All loading logic integrated directly - no use case objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from dlkit.common.shapes import InputShapes, OutputShapes

import torch
from tensordict import TensorDict

from dlkit.common.errors import ForwardContractError, WorkflowError
from dlkit.engine.adapters.lightning.base import _unpack_model_output
from dlkit.engine.adapters.lightning.functions import apply_inverse_chain
from dlkit.infrastructure.precision import (
    PrecisionService,
    get_precision_service,
    precision_override,
)
from dlkit.infrastructure.precision.strategy import PrecisionStrategy
from dlkit.infrastructure.utils.logging_config import get_logger

from .config import ModelState, PredictionOutput, PredictorConfig
from .loading import build_model_from_checkpoint, load_checkpoint
from .transforms import load_transforms_from_checkpoint

logger = get_logger(__name__)


def _move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Return a tensor placed on the requested device."""
    return tensor.to(device)


def _load_entry_shapes_from_checkpoint(
    checkpoint: dict,
) -> tuple[InputShapes | None, OutputShapes | None]:
    """Load entry input/output shapes serialized in checkpoint metadata.

    Args:
        checkpoint: Loaded checkpoint dict.

    Returns:
        Tuple of ``(input_shapes, output_shapes)``. Either element is ``None``
        when the corresponding shapes are absent (e.g. external checkpoints).
    """
    from dlkit.engine.adapters.lightning.concerns._checkpoint_serializer_helpers import (
        deserialize_shapes,
    )

    metadata = checkpoint.get("dlkit_metadata", {})
    input_shapes = deserialize_shapes(metadata.get("input_shapes"))
    output_shapes = deserialize_shapes(metadata.get("output_shapes"))
    if input_shapes is not None:
        logger.debug("Loaded entry shapes from checkpoint metadata: {}", input_shapes)
    return input_shapes, output_shapes


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
    ) -> PredictionOutput:
        """Execute inference, mirroring model.forward() signature."""
        ...

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names restored from checkpoint metadata."""
        ...

    def describe_inputs(self) -> dict[str, str]:
        """Return the checkpoint's persisted forward()-kwarg contract."""
        ...

    @property
    def predict_target_key(self) -> str:
        """Target key whose inverse transform applies to direct predictions."""
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
        >>> output1 = predictor.predict(x_tensor)
        >>> output2 = predictor.predict(x=x_tensor, edge_attr=ea_tensor)
        >>>
        >>> # Clean up
        >>> predictor.unload()

        >>> # Or with context manager
        >>> with CheckpointPredictor(config, auto_load=True) as predictor:
        ...     output = predictor.predict(x=inputs)
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

        # Load entry shapes from checkpoint metadata for entry-consumer reconstruction.
        logger.debug("Loading entry shapes from checkpoint metadata")
        input_shapes, output_shapes = _load_entry_shapes_from_checkpoint(checkpoint)

        # Build and load model
        logger.debug("Building model from checkpoint")
        model = build_model_from_checkpoint(checkpoint, input_shapes, output_shapes)

        # Load transforms (separated by type)
        logger.debug("Loading fitted transforms")
        feature_transforms, target_transforms = load_transforms_from_checkpoint(checkpoint)

        # Place model and transforms on device
        device = self._resolve_device()
        logger.debug("Moving model to device: {}", device)
        model = model.to(device)
        for chain in feature_transforms.values():
            chain.to(device)
        for chain in target_transforms.values():
            chain.to(device)

        # Ensure eval mode
        model.eval()

        # Extract feature_names, forward_arg_map, and predict_target_key from checkpoint
        meta = checkpoint.get("dlkit_metadata", {})
        raw_fn = meta.get("feature_names", ())
        feature_names: tuple[str, ...] = tuple(raw_fn) if isinstance(raw_fn, (list, tuple)) else ()
        predict_target_key: str = str(meta.get("predict_target_key", ""))
        # Empty dict = positional mode (old checkpoints); non-empty = named kwarg dispatch.
        raw_fam = meta.get("forward_arg_map", {})
        forward_arg_map: dict[str, str] = dict(raw_fam) if isinstance(raw_fam, dict) else {}

        # Create model state
        self._model_state = ModelState(
            model=model,
            device=device,
            feature_transforms=feature_transforms if feature_transforms else None,
            target_transforms=target_transforms if target_transforms else None,
            metadata=meta,
            feature_names=feature_names,
            predict_target_key=predict_target_key,
            forward_arg_map=forward_arg_map,
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
    ) -> PredictionOutput:
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
            PredictionOutput with the primary prediction tensor, optional latent
            tensors, and raw TensorDict output when the model returned one.

        Raises:
            PredictorNotLoadedError: If predictor not loaded.
        """
        if not self._loaded or self._model_state is None:
            raise PredictorNotLoadedError()

        self._validate_kwarg_names(kwargs)

        # Establish precision context (inferred from model)
        precision_to_use = self._config.precision or self._inferred_precision

        if precision_to_use is not None:
            ctx = precision_override(precision_to_use)
        else:
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            args, kwargs = self._move_inputs_to_device(args, kwargs)
            if precision_to_use is not None:
                args, kwargs = self._cast_inputs_to_precision(args, kwargs, precision_to_use)
            if self._config.apply_transforms:
                args, kwargs = self._apply_input_transforms(args, kwargs)

            with torch.no_grad():
                raw_output = self._model_state.model(*args, **kwargs)

            predictions_raw, latents_raw = _unpack_model_output(raw_output)

            if self._config.apply_transforms:
                predictions_raw = self._apply_output_inverse_transform(predictions_raw)

            if not isinstance(predictions_raw, torch.Tensor):
                raise TypeError(
                    "CheckpointPredictor.predict() requires the primary prediction output "
                    "to be a torch.Tensor. Return a Tensor directly, a tuple whose first "
                    "element is a Tensor, or a TensorDict with a 'predictions' tensor."
                )

            latents: tuple[torch.Tensor, ...]
            if latents_raw is None:
                latents = ()
            elif isinstance(latents_raw, tuple):
                latents = latents_raw
            else:
                latents = (latents_raw,)

            if not all(isinstance(latent, torch.Tensor) for latent in latents):
                raise TypeError(
                    "CheckpointPredictor.predict() requires all latent outputs to be "
                    "torch.Tensor instances."
                )

            raw = raw_output if isinstance(raw_output, TensorDict) else None
            return PredictionOutput(
                predictions=predictions_raw,
                latents=latents,
                raw=raw,
            )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names restored from checkpoint metadata."""
        if self._model_state is None:
            return ()
        return self._model_state.feature_names

    @property
    def predict_target_key(self) -> str:
        """Target key whose inverse transform applies to direct predictions."""
        if self._model_state is None:
            return ""
        return self._model_state.predict_target_key

    def describe_inputs(self) -> dict[str, str]:
        """Return the checkpoint's persisted forward()-kwarg contract.

        Maps each expected keyword argument name to the feature entry it is
        sourced from (identity today — see ``ModelState.forward_arg_map``).
        Empty dict for legacy/positional-mode checkpoints; call ``predict()``
        with positional args in ``feature_names`` order instead.

        Returns:
            Mapping of forward() kwarg name to feature entry name.
        """
        if self._model_state is None:
            return {}
        return dict(self._model_state.forward_arg_map)

    def _validate_kwarg_names(self, kwargs: dict[str, torch.Tensor]) -> None:
        """Fail fast with a dlkit-authored message on an unexpected kwarg name.

        Compares caller-supplied kwarg names against the checkpoint's
        persisted ``forward_arg_map`` before the model is ever called — the
        same failure point Python's own call machinery would already raise
        at for a genuinely wrong name, just with a message naming what the
        checkpoint actually expects instead of a raw ``TypeError`` from deep
        inside the model. Only checks kwargs; positional args are validated
        by Python's own call machinery, unchanged.

        Args:
            kwargs: Caller-supplied named tensor inputs.

        Raises:
            ForwardContractError: If any kwarg name is not in the checkpoint's
                persisted forward-arg contract.
        """
        assert self._model_state is not None
        expected = frozenset(self._model_state.forward_arg_map)
        if not expected:
            return  # legacy/positional-mode checkpoint — nothing to validate
        unknown = frozenset(kwargs) - expected
        if unknown:
            raise ForwardContractError(
                f"predict() received unexpected keyword argument(s) {sorted(unknown)}. "
                f"This checkpoint expects: {sorted(expected)}. Call describe_inputs() "
                "to inspect the full contract.",
                {"unexpected": sorted(unknown), "expected": sorted(expected)},
            )

    def _move_inputs_to_device(
        self,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        """Move caller-provided tensor inputs to the loaded predictor device."""
        if self._model_state is None:
            return args, kwargs

        device = torch.device(self._model_state.device)
        return (
            tuple(_move_tensor_to_device(arg, device) for arg in args),
            {key: _move_tensor_to_device(value, device) for key, value in kwargs.items()},
        )

    def _cast_inputs_to_precision(
        self,
        args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
        precision: PrecisionStrategy,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        """Cast caller-provided tensor inputs to the resolved precision dtype.

        Dataloader/raw-array tensors commonly arrive as float64 while
        checkpointed model parameters are float32 (or another inferred
        precision) — without this, the first matmul inside the model raises
        a dtype mismatch. Mirrors ``_move_inputs_to_device``'s device
        handling, but for dtype.
        """
        return (
            tuple(self._precision_service.cast_tensor(arg, default=precision) for arg in args),
            {
                key: self._precision_service.cast_tensor(value, default=precision)
                for key, value in kwargs.items()
            },
        )

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

        chain = tt[target_key]

        if isinstance(predictions, torch.Tensor):
            return apply_inverse_chain(predictions, chain)

        # TensorDict: apply inverse to its "predictions" leaf if present
        if isinstance(predictions, TensorDict) and "predictions" in predictions.keys():
            from typing import cast as _cast

            return _cast(
                TensorDict,
                predictions.apply(
                    lambda t: apply_inverse_chain(t, chain), batch_size=predictions.batch_size
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
