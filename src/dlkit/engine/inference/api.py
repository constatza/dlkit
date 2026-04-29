"""Inference API - Simplified Stateful Predictor Architecture.

Public API for creating and using predictors. Factory function pattern
without DI containers or factories.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.common import ConfigurationError
from dlkit.infrastructure.config.workflow_configs import InferenceWorkflowConfig
from dlkit.infrastructure.precision.strategy import PrecisionStrategy

from .config import PredictionOutput, PredictorConfig
from .loading import (
    CheckpointInfo,
    CheckpointValidationResult,
    get_checkpoint_info,
    validate_checkpoint,
)
from .predictor import CheckpointPredictor


def load_model(
    checkpoint_path: Path | str,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True,
    auto_load: bool = True,
    precision: PrecisionStrategy | None = None,
) -> CheckpointPredictor:
    """Load a stateful predictor from checkpoint (PRIMARY API).

    Simple factory function - no factory classes or DI containers.
    Creates and optionally loads a predictor in one call.

    This is the recommended way to perform inference. The predictor loads
    the model ONCE and can be reused for multiple predictions without
    reloading the checkpoint.

    For iterative workflows (100+ predictions), this provides 10-100x
    performance improvement compared to reloading on each call.

    Args:
        checkpoint_path: Path to trained model checkpoint
        device: Device specification:
            - "auto": Automatically select (CUDA > MPS > CPU)
            - "cpu": Force CPU
            - "cuda" or "cuda:0": Use specific CUDA device
            - "mps": Use Apple Silicon GPU
        batch_size: Default batch size for inference
        apply_transforms: Whether to apply fitted transforms from checkpoint
        auto_load: If True, load model immediately (recommended)
        precision: Optional precision override for data loading.
                  If None, infers precision from model checkpoint dtype.

    Returns:
        CheckpointPredictor: Reusable predictor object

    Raises:
        WorkflowError: If checkpoint loading fails

    Examples:
        >>> # Basic usage (one-shot inference)
        >>> from dlkit import load_model
        >>> with load_model("model.ckpt") as predictor:
        ...     output = predictor.predict(x=torch.randn(32, 10))
        ...     predictions = output.predictions

        >>> # Efficient multi-inference
        >>> predictor = load_model("model.ckpt", device="cuda")
        >>> for data in dataset:  # No reloading!
        ...     output = predictor.predict(x=data)
        ...     process(output.predictions)
        >>> predictor.unload()

        >>> # Lazy loading
        >>> predictor = load_model("model.ckpt", auto_load=False)
        >>> # ... do other setup ...
        >>> predictor.load()  # Explicit load
        >>> output = predictor.predict(x=data)
    """
    config = PredictorConfig(
        checkpoint_path=Path(checkpoint_path),
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms,
        auto_load=auto_load,
        precision=precision,
    )

    return CheckpointPredictor(config)


def load_model_from_settings(
    settings: InferenceWorkflowConfig,
    *,
    checkpoint_path: Path | str | None = None,
    device: str = "auto",
    batch_size: int = 32,
    apply_transforms: bool = True,
    auto_load: bool = True,
    precision: PrecisionStrategy | None = None,
) -> CheckpointPredictor:
    """Resolve a checkpoint from inference settings or an explicit override."""
    model_settings = settings.MODEL
    resolved_checkpoint = checkpoint_path or (
        model_settings.checkpoint if model_settings is not None else None
    )
    if resolved_checkpoint is None:
        raise ConfigurationError("No checkpoint path found in settings or override.")
    return load_model(
        checkpoint_path=resolved_checkpoint,
        device=device,
        batch_size=batch_size,
        apply_transforms=apply_transforms,
        auto_load=auto_load,
        precision=precision,
    )


# Re-export functions and dataclasses
__all__ = [
    "CheckpointInfo",
    "CheckpointValidationResult",
    "PredictionOutput",
    "get_checkpoint_info",
    "load_model",
    "load_model_from_settings",
    "validate_checkpoint",
]
