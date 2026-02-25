"""Inference API - Simplified Stateful Predictor Architecture.

Public API for creating and using predictors. Factory function pattern
without DI containers or factories.
"""

from __future__ import annotations

from pathlib import Path

from dlkit.tools.config.precision.strategy import PrecisionStrategy

from .config import PredictorConfig
from .predictor import CheckpointPredictor
from .loading import (
    validate_checkpoint,
    get_checkpoint_info,
    CheckpointValidationResult,
    CheckpointInfo,
)


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
        ...     result = predictor.predict({"x": torch.randn(32, 10)})

        >>> # Efficient multi-inference
        >>> predictor = load_model("model.ckpt", device="cuda")
        >>> for data in dataset:  # No reloading!
        ...     result = predictor.predict(data)
        ...     process(result)
        >>> predictor.unload()

        >>> # Lazy loading
        >>> predictor = load_model("model.ckpt", auto_load=False)
        >>> # ... do other setup ...
        >>> predictor.load()  # Explicit load
        >>> result = predictor.predict(data)
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


# Re-export functions and dataclasses
__all__ = [
    "load_model",
    "validate_checkpoint",
    "get_checkpoint_info",
    "CheckpointValidationResult",
    "CheckpointInfo",
]
